# ML Algorithms with NuMojo

This repository contains implementations of Machine Learning algorithms with NuMojo library as the backend. I created this repo for two main purposes,  

1. I want to implement Machine learning algorithms from scratch by myself and understand them.

2. As a developer of the NuMojo library, this project helps identify areas for improvement and missing features in NuMojo. By implementing practical ML algorithms, I can better understand what features are lacking and what could be improved in NuMojo.

## Current Implementations

- Linear Regression
- Logistic Regression

I will add more algorithms as I learn and implement. 

## Getting Started

This project uses Magic, the Mojo package manager. Follow these steps to get started:

1. Clone the repository:
   ```
   git clone https://github.com/shivasankarka/MLAlgorithms.git
   ```

2. Navigate to the project directory:
   ```
   cd MLAlgorithms
   ```

3. Run test files using Magic. For example, to test Linear Regression:
   ```
   magic shell
   cd tests
   mojo test_linearReg.mojo -I ../
   ```

## Contributing

Contributions are welcome! If you'd like to improve existing ones, or enhance documentation, please feel free to submit a pull request.

## Future Work

- Implement more ML algorithms (e.g., KNN, Decision Trees, Random Forests, SVM, KNN, Naive Bayes, Perceptron, etc)
- Optimize existing implementations for better performance
- Add more comprehensive testing and benchmarking

## Acknowledgements

This project is inspired by:
- [MojoMelo](https://github.com/yetalit/mojmelo)
- [MLfromscratch](https://github.com/patrickloeber/MLfromscratch)

Many of the test cases are adapted from [MojoMelo](https://github.com/yetalit/mojmelo).
