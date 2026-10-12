# Orbit CLI

Orbit is a command-line tool for syncing project folders to a self-hosted storage server. It is written in Go and ships as a single static binary.

## Installation

### From a release

Download the archive for your platform from the releases page and unpack it:

```bash
tar -xzf orbit-2.4.1-linux-amd64.tar.gz
sudo mv orbit /usr/local/bin/
orbit --version
```

The current stable release is **2.4.1**, published on 3 March 2026.

### With a package manager

- Homebrew: `brew install orbit-sync/tap/orbit`
- Scoop: `scoop install orbit`
- Arch Linux: the package `orbit-bin` is maintained in the AUR.

## Configuration

Orbit reads its settings from `~/.config/orbit/orbit.toml`. A minimal file looks like this:

```toml
server   = "https://files.example.net"
port     = 7443
workers  = 6
compress = "zstd"
```

| Key        | Default | Meaning                                   |
|------------|---------|-------------------------------------------|
| `port`     | `7443`  | TCP port of the storage server            |
| `workers`  | `4`     | Number of parallel upload workers         |
| `compress` | `none`  | Compression codec: `none`, `gzip`, `zstd` |
| `retries`  | `5`     | Attempts before a transfer is marked failed |

Environment variables override the file. For example, `ORBIT_WORKERS=8` sets eight workers for one run.

## Commands

### orbit push

Uploads every changed file in the current folder. Files listed in `.orbitignore` are skipped. Use `--dry-run` to print the plan without transferring anything.

### orbit pull

Downloads remote changes. When the same file changed on both sides, Orbit keeps the remote copy and saves the local one as `name.conflict`.

### orbit status

Shows pending uploads and downloads. The flag `--json` prints machine-readable output.

## Troubleshooting

### Error E104: certificate rejected

The server uses a certificate that your system does not trust. Add the certificate path to the config with `ca_file = "/path/to/ca.pem"`. Do not disable verification.

### Transfers are slow

Raise `workers` gradually; values above 16 rarely help. Enabling `compress = "zstd"` speeds up text-heavy projects by about 40 percent in our benchmarks.

## License

Orbit is released under the Mozilla Public License 2.0. Contributions require signing the contributor agreement in `CONTRIBUTING.md`.
