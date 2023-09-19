# Wave Function Collapse

This is my attempt at implementing [the Wave Function Collapse algorithm](https://github.com/mxgmn/WaveFunctionCollapse) in Rust.

## Some Context

I'm fairly new to Rust, I started this project to get more experience with the
language, and it has allowed me to touch on almost all main aspects of the
language.

## Some Notes

**This implementation is still in its infancy and the code is far from perfect**,
but it does do *some* things right.

1. Works in **any dimension** from 1 to 6. (*dimensions for which `ndarray` has
   a `const N: usize` where `Dim<[usize; N]>`* is an implemented type)

   **Note**: Only tested in 2D for now.

2. Works on **any type** that implements the simple `Hashable` trait.
3. Type safety at compile time! Type conditions on the wave are checked at
   compile time, so no surprises at runtime.
4. **Parallelism**! The wave propagation step is parallelized out of the box
   using `rayon` resulting in `4x` improvements on my minimal testing so far

While this all sounds very nice, this project is not perfect. Here are some
things I still need to work on.

1. `Tiles` don't own their data.

    This is great if your tiles come from a sliding window on an image, but not
    great when you want to *supply* them directly.

2. Backtracking isn't implemented yet.

    Although the datastructures allow it, backtracking isn't added yet resulting in some waves with holes in them.

4. The code has lots of clones, and generally not the best at the moment.

    This shouldn't be hard to fix though, now that the main pillars are in place.

5. The wave can be stopped early when no more change is required.

    This will probably be a fairly large, free improvement once its written.

6. Better docs is required.

## Getting Started

You might need sdl binaries for you system if you want to display the wave.
Other than that

```
cargo run -- ./assets/sample.png 10 3x3
```

should get you some results!
