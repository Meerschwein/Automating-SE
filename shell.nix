let
  pkgs = import <nixpkgs> {system = "x86_64-linux";};
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      (python3.withPackages (ps:
        with ps; [
          pandas
          scikit-learn
        ]))
    ];
  }
