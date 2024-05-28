# Wave Function Collapse

An implementation of the [Wave Function Collapse](https://github.com/mxgmn/WaveFunctionCollapse)
algorithm written in Rust. This implementation strives to be as general as
possible while staying as fast as possible.

## Notable Features

- Works in any dimension from 1-6
- Works on any type which implements `WaveTileable`
- Wave can be collapsed in parallel for types which are also `Send` and `Sync`

## Feature Flags

The crate comes with a set of useful (and common) features that deal with wave
function collapse.

- `image` (default) Allows the user to easily crate and collapse `Wave`s of images.
- `parallel` (default) Allows the user to collapse a `Wave` in parallel for types which are `Send` and `Sync`.
- `wrapping` Allows the user to create a wrapping `Wave`. Currently, supported wrapping modes are
    - `Torus`
    - `ProjectivePlane`
    - `KleinBottle`

## Getting Started

The following command runs the `image` example, with the `rooms.png` sample on a
tiling window of sidelength 3.

```
cargo run --release --example image -- ./examples/image/samples/rooms.png 3
```
