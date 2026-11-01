use std::sync::Arc;

use bytes::Bytes;
use futures::{stream, StreamExt, TryStreamExt};

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::heap::local::LocalHeap;
use crate::reader::AsyncFileReader;

/// A single entry from a v1 B-tree group node (type 0).
///
/// Maps a link name to a symbol table node address.
#[derive(Debug, Clone)]
pub struct SymbolTableEntry {
    /// Link name (resolved from local heap).
    pub name: String,
    /// File address of the child object's header.
    pub object_header_address: u64,
    /// Cache type (0=none, 1=group, 2=symlink).
    pub cache_type: u32,
    /// For cache_type=1: child B-tree address (from scratch-pad).
    pub child_btree_address: Option<u64>,
    /// For cache_type=1: child local heap address (from scratch-pad).
    pub child_heap_address: Option<u64>,
}

/// A v1 B-tree node for raw data chunks (type 1).
///
/// Each entry maps chunk coordinates to a byte offset and size in the file.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    /// Size of the chunk in bytes (compressed/on-disk size).
    pub size: u32,
    /// Filter mask — bit N set means filter N was *not* applied.
    pub filter_mask: u32,
    /// Chunk offsets in each dimension.
    pub offsets: Vec<u64>,
    /// File address of the chunk data.
    pub address: u64,
}

/// Traverse a v1 B-tree (type 0 — group nodes) and collect all symbol table entries.
///
/// This recursively follows the tree from the given root address, eventually
/// reaching leaf nodes that point to Symbol Table Nodes (SNOD). Each SNOD
/// contains symbol table entries mapping names (via the local heap) to object
/// header addresses.
pub async fn read_group_btree_v1(
    reader: &Arc<dyn AsyncFileReader>,
    btree_address: u64,
    local_heap: &LocalHeap,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<SymbolTableEntry>> {
    let mut entries = Vec::new();
    read_group_btree_node(
        reader,
        btree_address,
        local_heap,
        size_of_offsets,
        size_of_lengths,
        &mut entries,
    )
    .await?;
    Ok(entries)
}

/// Recursively read a v1 B-tree node.
async fn read_group_btree_node(
    reader: &Arc<dyn AsyncFileReader>,
    address: u64,
    local_heap: &LocalHeap,
    size_of_offsets: u8,
    size_of_lengths: u8,
    entries: &mut Vec<SymbolTableEntry>,
) -> Result<()> {
    // Fetch enough data for the node header + reasonable number of entries.
    // Node header: 4 (sig) + 1 (type) + 1 (level) + 2 (entries_used) + 2*O (siblings)
    // Key size for type 0 = L bytes. Each entry = key(L) + child(O).
    // Pattern: key[0], child[0], key[1], child[1], ..., key[N-1], child[N-1], key[N]
    // Total keys = entries_used + 1, total children = entries_used.
    // Conservative fetch: 8KB should cover most nodes.
    let fetch_size = 8192u64;
    let data = reader.get_bytes(address..address + fetch_size).await?;
    let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

    // Verify signature
    r.read_signature(b"TREE")?;

    let node_type = r.read_u8()?;
    if node_type != 0 {
        return Err(HDF5Error::General(format!(
            "expected B-tree v1 node type 0 (group), got {node_type}"
        )));
    }

    let node_level = r.read_u8()?;
    let entries_used = r.read_u16()? as usize;
    let _left_sibling = r.read_offset()?;
    let _right_sibling = r.read_offset()?;

    if node_level > 0 {
        // Internal node: keys and child pointers alternate.
        // key[0], child[0], key[1], child[1], ..., key[N-1], child[N-1], key[N]
        for _ in 0..entries_used {
            let _key = r.read_length()?; // heap offset (type 0 key)
            let child_address = r.read_offset()?;

            // Recurse into child node
            Box::pin(read_group_btree_node(
                reader,
                child_address,
                local_heap,
                size_of_offsets,
                size_of_lengths,
                entries,
            ))
            .await?;
        }
        // Skip trailing key
        let _trailing_key = r.read_length()?;
    } else {
        // Leaf node: child pointers are addresses of Symbol Table Nodes (SNOD).
        let mut child_addresses = Vec::with_capacity(entries_used);
        for _ in 0..entries_used {
            let _key = r.read_length()?;
            let child_address = r.read_offset()?;
            child_addresses.push(child_address);
        }
        // Skip trailing key
        let _trailing_key = r.read_length()?;

        // Read each SNOD
        for snod_address in child_addresses {
            let snod_entries = read_symbol_table_node(
                reader,
                snod_address,
                local_heap,
                size_of_offsets,
                size_of_lengths,
            )
            .await?;
            entries.extend(snod_entries);
        }
    }

    Ok(())
}

/// Read a Symbol Table Node (SNOD) and extract all its entries.
///
/// Binary layout:
///   - Signature "SNOD" (4 bytes)
///   - Version (1 byte): 1
///   - Reserved (1 byte)
///   - Number of Symbols (2 bytes)
///   - Entries... (each: name_offset(O) + header_address(O) + cache_type(4) + reserved(4) + scratch(16))
async fn read_symbol_table_node(
    reader: &Arc<dyn AsyncFileReader>,
    address: u64,
    local_heap: &LocalHeap,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<SymbolTableEntry>> {
    // Each symbol table entry: 2*O + 4 + 4 + 16 = 2*O + 24 bytes.
    // Max 2K entries (K from superblock, typically 16-32).
    // Conservative: fetch 16KB.
    let fetch_size = 16384u64;
    let data = reader.get_bytes(address..address + fetch_size).await?;
    let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

    // Verify signature
    r.read_signature(b"SNOD")?;

    let version = r.read_u8()?;
    if version != 1 {
        return Err(HDF5Error::General(format!(
            "unsupported SNOD version: {version}"
        )));
    }

    let _reserved = r.read_u8()?;
    let num_symbols = r.read_u16()? as usize;

    let mut entries = Vec::with_capacity(num_symbols);

    for _ in 0..num_symbols {
        let name_offset = r.read_offset()?;
        let object_header_address = r.read_offset()?;
        let cache_type = r.read_u32()?;
        let _reserved = r.read_u32()?;

        // 16 bytes scratch-pad space
        let (child_btree_address, child_heap_address) = if cache_type == 1 {
            // Group cache: B-tree address + heap address
            let btree = r.read_offset()?;
            let heap = r.read_offset()?;
            // Skip remaining scratch-pad bytes (16 - 2*O)
            let used = 2 * size_of_offsets as u64;
            if used < 16 {
                r.skip(16 - used);
            }
            (Some(btree), Some(heap))
        } else {
            r.skip(16); // unused scratch-pad
            (None, None)
        };

        let name = local_heap.get_string(name_offset)?;

        entries.push(SymbolTableEntry {
            name,
            object_header_address,
            cache_type,
            child_btree_address,
            child_heap_address,
        });
    }

    Ok(entries)
}

/// Index node reads in flight during a chunk B-tree walk, unless a caller
/// sets another value.
pub const DEFAULT_CHUNK_BTREE_CONCURRENCY: usize = 32;

/// Traverse a v1 B-tree (type 1 — raw data chunks) and collect all chunk entries.
///
/// `ndims` is the dimensionality of the dataset (needed to parse chunk keys).
/// Reads [`DEFAULT_CHUNK_BTREE_CONCURRENCY`] nodes at a time; see
/// [`read_chunk_btree_v1_with_concurrency`].
pub async fn read_chunk_btree_v1(
    reader: &Arc<dyn AsyncFileReader>,
    btree_address: u64,
    ndims: usize,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<ChunkEntry>> {
    read_chunk_btree_v1_with_concurrency(
        reader,
        btree_address,
        ndims,
        size_of_offsets,
        size_of_lengths,
        DEFAULT_CHUNK_BTREE_CONCURRENCY,
    )
    .await
}

/// Traverse a v1 chunk B-tree one level at a time, reading up to
/// `concurrency` nodes of a level at once.
///
/// A file written one record at a time (the usual netCDF-4 time series)
/// spreads its index nodes through the whole file, so a walk that waits for
/// each node pays one round trip per node. Walking by level pays one round
/// trip per level. Entries come back in key order, the same as a
/// depth-first walk.
///
/// Each node is read at its own size: a full node with K = 32 (libhdf5's
/// default), read again whole if its entry count says it is bigger.
pub async fn read_chunk_btree_v1_with_concurrency(
    reader: &Arc<dyn AsyncFileReader>,
    btree_address: u64,
    ndims: usize,
    size_of_offsets: u8,
    size_of_lengths: u8,
    concurrency: usize,
) -> Result<Vec<ChunkEntry>> {
    // Key size for type 1: 4 (size) + 4 (filter_mask) + (ndims+1)*8 (offsets)
    let key_size = 8 + (ndims + 1) * 8;
    let offset_size = size_of_offsets as usize;
    let header_size = 8 + 2 * offset_size;
    let node_size =
        |entries: usize| (header_size + (entries + 1) * key_size + entries * offset_size) as u64;
    let guess = node_size(64);

    let mut level = vec![btree_address];
    let mut entries = Vec::new();
    let mut depth = 0;
    while !level.is_empty() {
        // Level 0 is a leaf; a tree deeper than this is a corrupt file.
        if depth > 64 {
            return Err(HDF5Error::General(
                "chunk B-tree deeper than 64 levels".into(),
            ));
        }
        let nodes: Vec<Bytes> = stream::iter(level)
            .map(|address| async move {
                let data = reader.get_bytes(address..address + guess).await?;
                if data.len() >= header_size {
                    let used = u16::from_le_bytes([data[6], data[7]]) as usize;
                    if node_size(used) > data.len() as u64 {
                        return reader.get_bytes(address..address + node_size(used)).await;
                    }
                }
                Ok(data)
            })
            .buffered(concurrency.max(1))
            .try_collect()
            .await?;

        let mut next = Vec::new();
        for data in nodes {
            let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);
            r.read_signature(b"TREE")?;

            let node_type = r.read_u8()?;
            if node_type != 1 {
                return Err(HDF5Error::General(format!(
                    "expected B-tree v1 node type 1 (chunk), got {node_type}"
                )));
            }

            let node_level = r.read_u8()?;
            let entries_used = r.read_u16()? as usize;
            let _left_sibling = r.read_offset()?;
            let _right_sibling = r.read_offset()?;

            for _ in 0..entries_used {
                if node_level > 0 {
                    // Internal node: skip the key, keep the child.
                    r.skip(key_size as u64);
                    next.push(r.read_offset()?);
                    continue;
                }
                // Leaf: size(4) + filter_mask(4) + offsets((ndims+1) * 8)
                let chunk_size = r.read_u32()?;
                let filter_mask = r.read_u32()?;
                let mut offsets = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    offsets.push(r.read_u64()?);
                }
                let _zero = r.read_u64()?; // trailing dimension (always 0)

                let child_address = r.read_offset()?;

                // Skip entries with undefined address (unallocated chunks)
                if !HDF5Reader::is_undef_addr(child_address, size_of_offsets) {
                    entries.push(ChunkEntry {
                        size: chunk_size,
                        filter_mask,
                        offsets,
                        address: child_address,
                    });
                }
            }
        }
        level = next;
        depth += 1;
    }
    Ok(entries)
}
