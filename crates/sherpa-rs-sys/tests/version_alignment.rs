use std::{fs, path::PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn parse_cmake_version(contents: &str) -> String {
    const PREFIX: &str = "set(SHERPA_ONNX_VERSION \"";

    contents
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix(PREFIX)
                .and_then(|rest| rest.strip_suffix("\")"))
                .map(str::to_owned)
        })
        .expect("SHERPA_ONNX_VERSION must be present in sherpa-onnx/CMakeLists.txt")
}

fn parse_dist_tag(contents: &str) -> String {
    let bytes = contents.as_bytes();
    let mut index = 0;
    let mut depth = 0;

    while let Some(&byte) = bytes.get(index) {
        match byte {
            b'"' => {
                let (key, next_index) = parse_json_string(bytes, index);

                if depth == 1 && key == "tag" {
                    let colon_index = skip_whitespace(bytes, next_index);
                    assert_eq!(bytes.get(colon_index), Some(&b':'));

                    let value_index = skip_whitespace(bytes, colon_index + 1);
                    let (value, _) = parse_json_string(bytes, value_index);
                    return value;
                }

                index = next_index;
            }
            b'{' | b'[' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' => {
                depth -= 1;
                index += 1;
            }
            _ => {
                index += 1;
            }
        }
    }

    panic!("dist.json must contain a top-level tag field")
}

fn parse_json_string(bytes: &[u8], start: usize) -> (String, usize) {
    assert_eq!(bytes.get(start), Some(&b'"'));

    let mut index = start + 1;
    let mut escaped = false;

    while let Some(&byte) = bytes.get(index) {
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            return (
                String::from_utf8(bytes[start + 1..index].to_vec())
                    .expect("dist.json strings should be valid UTF-8"),
                index + 1,
            );
        }

        index += 1;
    }

    panic!("dist.json contains an unterminated JSON string")
}

fn skip_whitespace(bytes: &[u8], mut index: usize) -> usize {
    while matches!(bytes.get(index), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        index += 1;
    }

    index
}

fn parse_configured_archives(contents: &str, tag: &str) -> Vec<String> {
    let bytes = contents.as_bytes();
    let mut index = 0;
    let mut archives = Vec::new();

    while let Some(&byte) = bytes.get(index) {
        if byte != b'"' {
            index += 1;
            continue;
        }

        let (key, next_index) = parse_json_string(bytes, index);
        let colon_index = skip_whitespace(bytes, next_index);

        if !matches!(key.as_str(), "static" | "dynamic" | "archive")
            || bytes.get(colon_index) != Some(&b':')
        {
            index = next_index;
            continue;
        }

        let value_index = skip_whitespace(bytes, colon_index + 1);
        if bytes.get(value_index) != Some(&b'"') {
            index = value_index;
            continue;
        }

        let (value, value_end) = parse_json_string(bytes, value_index);
        archives.push(value.replace("{tag}", tag));
        index = value_end;
    }

    archives.sort();
    archives.dedup();
    archives
}

#[test]
fn vendored_release_metadata_matches_vendored_sherpa_version() {
    let root = manifest_dir();
    let cmake = fs::read_to_string(root.join("sherpa-onnx/CMakeLists.txt"))
        .expect("should read vendored sherpa-onnx CMakeLists.txt");
    let dist = fs::read_to_string(root.join("dist.json"))
        .expect("should read crates/sherpa-rs-sys/dist.json");
    let checksum = fs::read_to_string(root.join("checksum.txt"))
        .expect("should read crates/sherpa-rs-sys/checksum.txt");

    let version = parse_cmake_version(&cmake);
    let dist_tag = parse_dist_tag(&dist);
    let configured_archives = parse_configured_archives(&dist, &dist_tag);

    assert_eq!(
        dist_tag,
        format!("v{version}"),
        "dist.json tag must match the vendored sherpa-onnx version"
    );

    let mismatches: Vec<&str> = checksum
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.contains(&version))
        .take(5)
        .collect();

    assert!(
        mismatches.is_empty(),
        "checksum.txt entries must all reference version {version}; first mismatches: {mismatches:?}"
    );

    let checksum_entries: Vec<&str> = checksum
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter_map(|line| line.split_once('\t').map(|(name, _)| name))
        .collect();
    let missing_archives: Vec<String> = configured_archives
        .iter()
        .filter(|archive| {
            !checksum_entries
                .iter()
                .any(|entry| entry == &archive.as_str())
        })
        .cloned()
        .collect();

    assert!(
        !configured_archives.is_empty(),
        "dist.json should configure at least one release archive"
    );
    assert!(
        missing_archives.is_empty(),
        "dist.json archives must all exist in checksum.txt; missing: {missing_archives:?}"
    );
}
