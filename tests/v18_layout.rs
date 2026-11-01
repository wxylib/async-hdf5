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
