# Wave Function Collapse

An implementation of the [Wave Function Collapse](https://github.com/mxgmn/WaveFunctionCollapse)
algorithm written in Rust. This implementation strives to be as general as
possible while maintaining optimum speed.

## Feature Flags

The crate comes with a set of useful (and common) features that deal with wave
function collapse.

- `image` (default) Allows the user to easily crate and collapse `Wave`s of images.
- `parallel` (WIP) Allows the user to collapse a `Wave` in parallel for types which are `Send` and `Sync`.
- `wrapping` Allows the user to create a wrapping `Wave`. Currently, supported wrapping modes are
    - `Torus`
    - `ProjectivePlane`
    - `KleinBottle`

## Getting Started

The following command runs the `image` example, with the `rooms.png` sample on a
tiling window of widelength 3.

```
cargo run --release --example image -- ./examples/image/samples/rooms.png 3
```
