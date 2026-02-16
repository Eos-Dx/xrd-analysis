fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shared_proto = "../../protocol/hub/v1/hub.proto";
    let shared_include = "../../protocol";
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(&[shared_proto], &[shared_include])?;
    println!("cargo:rerun-if-changed={shared_proto}");
    Ok(())
}
