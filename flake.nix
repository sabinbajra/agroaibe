{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      with pkgs;
      {
        devShells.default = mkShell rec {
          BIOME_BINARY = "${biome}/bin/biome";
          buildInputs = [
            nodejs_24
            pnpm
            # see https://github.com/biomejs/biome-vscode/issues/295
            # nixos can't deal with statically linked binaries, so
            # we need to use the biome nix package
            biome
          ];
        };
      }
    );
}
