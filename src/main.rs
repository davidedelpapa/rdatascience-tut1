use ndarray::prelude::*;
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand_distr::{Uniform, StandardNormal};
use ndarray_rand::rand as rand;
use rand::seq::IteratorRandom;
use ndarray_stats::HistogramExt;
use ndarray_stats::histogram::{strategies::Sqrt, GridBuilder};
use noisy_float::types::{N64, n64};


fn main() {
    // 1D array VS 1D array VS 1D Vec
    println!("\nCREATION\n");
    let arr1 = array![1., 2., 3., 4., 5., 6.];
    println!("1D array: \t{}", arr1);

    let ls1 = [1., 2., 3., 4., 5., 6.];
    println!("1D list: \t{:?}", ls1);

    let vec1 = vec![1., 2., 3., 4., 5., 6.];
    println!("1D vector: \t{:?}", vec1);

    // 1D array sums
    println!("\nSUM\n");

    let arr2 = array![1., 2.2, 3.3, 4., 5., 6.];
    let arr3 = arr1 + arr2;
    println!("1D array: \t{}", arr3);

    let ls2 = [1., 2.2, 3.3, 4., 5., 6.];
    let mut ls3 = ls1.clone();
    for i in 1..ls2.len(){
        ls3[i] = ls1[i] + ls2[i];
    }
    println!("1D list: \t{:?}", ls3);

    let vec2 = vec![1., 2.2, 3.3, 4., 5., 6.];
    let vec3: Vec<f64> = vec1.iter().zip(vec2.iter()).map(|(&e1, &e2)| e1 + e2).collect();
    println!("1D vec: \t{:?}", vec3);

    // Other kinds of arrays and operations
    println!("\nOTHER KINDS\n");


    let arr4 = array![[1., 2., 3.], [ 4., 5., 6.]];
    let arr5 = Array::from_elem((2, 1), 1.);
    let arr6 = arr4 + arr5;
    println!("2D array:\n{}", arr6);

    let arr7 = 	Array::<f64, _>::zeros(arr6.raw_dim());
    let arr8 = arr6 * arr7;
    println!("\n{}", arr8);

    // Identity matrix
    let identity: &Array2<f64> = &Array::eye(3);
    println!("\n{}", identity);

    let arr9 = &array![[1., 2., 3.], [ 4., 5., 6.], [7., 8., 9.]];
    let arr10 = arr9 * identity; // element-wise multiplication
    println!("\n{}", arr10);

    let arr11 = arr9.dot(identity); // proper matrix multiplication
    println!("\n{}", arr11);

    println!("\n{}", array![2.]); // 0-D array
    println!("Dimensions: {}", array![2.].ndim());

    //4D array

    let arr12 = Array::<i8, _>::ones((2, 3, 2, 2));
    println!("\nMULTIDIMENSIONAL\n{}", arr12);

    //ndarray-rand
    println!("\nRANDOM STUFF\n");

    let arr13 = Array::random((2, 5), Uniform::new(0., 10.));
    println!("{:5.2}", arr13);

    // Sampling
    let arr14 = array![1., 2., 3., 4., 5., 6.];
    let arr15 = arr14.sample_axis(Axis(0), 2, SamplingStrategy::WithoutReplacement);
    println!("\nSampling from:\t{}\nTwo elements:\t{}", arr14, arr15);

    // Sampling throug regular rand crate and Array::from_shape_vec
    let mut rng = rand::thread_rng();
    let faces = "😀😎😐😕😠😢";
    let arr16 = Array::from_shape_vec((2, 2), faces.chars().choose_multiple(&mut rng, 4)).unwrap();
    println!("\nSampling from:\t{}", faces);
    println!("Elements:\n{}", arr16);


    // DATAVIZ
    
    // Histogram
    
    let arr17 = Array::<f64, _>::random_using((10000, 2), StandardNormal, &mut rand::thread_rng());

    let data = arr17.mapv(|e| n64(e));
    let grid = GridBuilder::<Sqrt<N64>>::from_array(&data).unwrap().build();
    let histogram = data.histogram(grid);

    let histogram_matrix = histogram.counts();
    
    let data = histogram_matrix.sum_axis(Axis(0));
    let his_data: Vec<(f32, f32)> = data.iter().enumerate().map(|(e, i)| (e as f32, *i as f32) ).collect();
    
    let file = std::fs::File::create("standard_normal_hist.svg").unwrap(); // create file on disk
    
    let mut graph = poloto::plot("Histogram", "x", "y"); // create graph
    graph.histogram("Stand.Norm.Dist.", his_data).ymarker(0);
    
    graph.simple_theme(poloto::upgrade_write(file));
    

    // Scatter Plot

    let arr18 = Array::<f64, _>::random_using((300, 2), StandardNormal, &mut rand::thread_rng());

    let data:  Vec<(f64, f64)> = arr18.axis_iter(Axis(0)).map(|e| {
        let v = e.to_vec();
        (v[0], v[1])
    }).collect();

    let x_line = [[-3,0], [3,0]];
    let y_line = [[0,-3], [0, 3]];
    
    let file = std::fs::File::create("standard_normal_scatter.svg").unwrap(); // create file on disk
    
    let mut graph = poloto::plot("Scatter Plot", "x", "y"); // create graph
    graph.line("", &x_line);
    graph.line("", &y_line);
    graph.scatter("Stand.Norm.Dist.", data).ymarker(0);
    
    graph.simple_theme(poloto::upgrade_write(file));   
}
