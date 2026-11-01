//! Files with the layout netCDF-4 files usually have (libver v108):
//! version 3 attribute messages, dense attribute storage and version 1
//! B-tree chunk indexes. Made by `tests/fixtures/make_v18_fixtures.py`.

use std::path::PathBuf;

use async_hdf5::reader::TokioReader;
use async_hdf5::{AttributeValue, HDF5File};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

async fn open_fixture(name: &str) -> HDF5File {
    let file = tokio::fs::File::open(fixture_path(name)).await.unwrap();
    HDF5File::open(TokioReader::new(file)).await.unwrap()
}

#[tokio::test]
async fn version_3_attribute_messages() {
    let file = open_fixture("v18_layout.h5").await;
    let ds = file
        .root_group()
        .await
        .unwrap()
        .dataset("small")
        .await
        .unwrap();
    let mut names: Vec<String> = ds.attributes().await.into_iter().map(|a| a.name).collect();
    names.sort();
    assert_eq!(names, vec!["scale_factor", "units"]);
    let units = ds.attribute("units").await.unwrap();
    assert_eq!(units.value.as_str(), Some("m"));
    let scale = ds.attribute("scale_factor").await.unwrap();
    assert!(matches!(scale.value, AttributeValue::F32(ref v) if v == &[0.5]));
}

#[tokio::test]
async fn dense_attributes() {
    let file = open_fixture("v18_layout.h5").await;
    let ds = file
        .root_group()
        .await
        .unwrap()
        .dataset("dense")
        .await
        .unwrap();
    let attrs = ds.attributes().await;
    assert_eq!(attrs.len(), 13);
    for i in 0..12 {
        let a = ds.attribute(&format!("attr_{i:02}")).await.unwrap();
        assert_eq!(a.value.as_i32(), Some(i));
    }
    let long_name = ds.attribute("long_name").await.unwrap();
    assert_eq!(long_name.value.as_str(), Some("dense attribute storage"));
}

/// libhdf5's chunk list (H5Dget_chunk_info), written next to the fixture.
fn grid_oracle() -> Vec<(Vec<u64>, u32, u64, u64)> {
    let csv = std::fs::read_to_string(fixture_path("v18_layout.grid.csv")).unwrap();
    csv.lines()
        .skip(1)
        .map(|l| {
            let f: Vec<u64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            (vec![f[0], f[1]], f[2] as u32, f[3], f[4])
        })
        .collect()
}

#[tokio::test]
async fn version_1_chunk_btree_matches_libhdf5_at_any_concurrency() {
    let oracle = grid_oracle();
    assert_eq!(oracle.len(), 400);
    for concurrency in [1, 3, 32] {
        let file = open_fixture("v18_layout.h5").await;
        let ds = file
            .root_group()
            .await
            .unwrap()
            .dataset("grid")
            .await
            .unwrap()
            .with_index_concurrency(concurrency);
        let index = ds.chunk_index().await.unwrap();
        // Chunk offsets are element offsets; with 1 x 1 chunks they are
        // also the chunk indices.
        let got: Vec<(Vec<u64>, u32, u64, u64)> = index
            .iter()
            .map(|c| {
                (
                    c.indices.clone(),
                    c.filter_mask,
                    c.byte_offset,
                    c.byte_length,
                )
            })
            .collect();
        assert_eq!(got, oracle, "concurrency {concurrency}");
    }
}
