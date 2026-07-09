use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_URL: &str =
    "https://d1ruypri5fhwvl.cloudfront.net/felatab/v1/felatab_int8.safetensors";
const DEFAULT_SHA256: &str = "547c451a182f4a61aa4ab811efd1fe2e8c57b75b61e4256b28114934dc539741";

fn main() {
    println!("cargo:rerun-if-env-changed=FELATAB_WEIGHTS");
    println!("cargo:rerun-if-env-changed=FELATAB_MODEL_URL");
    println!("cargo:rerun-if-env-changed=FELATAB_MODEL_SHA256");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let dst = PathBuf::from(&out_dir).join("felatab_int8.safetensors");
    let want_sha = std::env::var("FELATAB_MODEL_SHA256").unwrap_or_else(|_| DEFAULT_SHA256.into());

    if dst.exists() && sha256_of(&dst).as_deref() == Some(want_sha.as_str()) {
        return;
    }

    if let Ok(local) = std::env::var("FELATAB_WEIGHTS") {
        println!("cargo:rerun-if-changed={local}");
        let src = Path::new(&local);
        if !src.exists() {
            fail(&format!(
                "FELATAB_WEIGHTS is set to \"{local}\" but that file does not exist. Point it at a \
                 local felatab_int8.safetensors, or unset it to download from FELATAB_MODEL_URL."
            ));
        }
        std::fs::copy(src, &dst).unwrap_or_else(|e| {
            fail(&format!(
                "cannot copy FELATAB_WEIGHTS \"{local}\" -> {dst:?}: {e}"
            ))
        });
    } else {
        let url = std::env::var("FELATAB_MODEL_URL").unwrap_or_else(|_| DEFAULT_URL.into());
        download(&url, &dst);
    }

    let got = sha256_of(&dst).unwrap_or_else(|| fail("sha256 tool (sha256sum/shasum) not found"));
    if got != want_sha {
        let _ = std::fs::remove_file(&dst);
        fail(&format!(
            "model sha256 mismatch for {dst:?}: got {got}, want {want_sha}. The download or local \
             FELATAB_WEIGHTS file is wrong/corrupt; refetch it or set FELATAB_MODEL_SHA256 to match."
        ));
    }
}

fn download(url: &str, dst: &Path) {
    let status = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(dst)
        .arg(url)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => fail(&format!(
            "curl failed ({s}) downloading model from {url}. Set FELATAB_WEIGHTS to a local \
             felatab_int8.safetensors for an offline build, or fix FELATAB_MODEL_URL / network."
        )),
        Err(e) => fail(&format!(
            "cannot run curl to download {url}: {e}. Install curl, or set FELATAB_WEIGHTS to a \
             local felatab_int8.safetensors for an offline build."
        )),
    }
}

fn sha256_of(path: &Path) -> Option<String> {
    for (bin, args) in [("sha256sum", &[][..]), ("shasum", &["-a", "256"][..])] {
        if let Ok(out) = Command::new(bin).args(args).arg(path).output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                if let Some(hex) = s.split_whitespace().next() {
                    return Some(hex.to_ascii_lowercase());
                }
            }
        }
    }
    None
}

fn fail(msg: &str) -> ! {
    panic!("pg_fela build.rs: {msg}");
}
