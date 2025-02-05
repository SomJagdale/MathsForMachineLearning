## Module I
### Relationship between machine learning, leaner algebra, and vector and matrices.
### Vectors 
### Vector addition, Vector substraction and vector scalling

## Module II
### Modulus and inner Product - Properties -- Commutitive, distributive over addition, associative over scalar multiplication
### Cosin and dot product 
### Projectin - Scalar projection and Vector projection
### Dot product of vectors 
Purpose in Machine Learning

**Scalar Projection:**
Helps in understanding how much one vector contributes to another, which can be useful in feature selection and dimensionality reduction.

**Vector Projection:**
Useful in algorithms that require directionality, such as gradient descent, where understanding the direction of the steepest descent is crucial for optimization.

Applications of the dot product (also called the scalar product)
p=vi cos0
W=fd cos0
Cosine Similarity in NLP & Data Science:

cosθ= A⋅B/∣A∣∣B∣
 
Measures similarity between text documents and features vectors in recommendation systems.
Projection of One Vector onto Another: Used to find how much of one vector lies along another.
Angle Between Two Vectors: Essential in navigation, physics, and structural analysis.


**Basis Function**
A basis function is a fundamental building block used to represent more complex functions in terms of simpler, well-defined components. It comes up in linear algebra, signal processing, machine learning, and numerical methods.
A **basis function** is a fundamental building block used to represent more complex functions in terms of simpler, well-defined components. It comes up in **linear algebra, signal processing, machine learning, and numerical methods**.  

### **Why is it Important?**  
In many fields, we approximate or decompose a function using a **linear combination** of basis functions. These basis functions are usually chosen to make computations easier.

### **Examples of Basis Functions**  

1. **Fourier Series (Signal Processing & PDEs)**  
   - Uses **sine and cosine** as basis functions:  
     \[
     f(x) = a_0 + \sum_{n=1}^{\infty} \left( a_n \cos(nx) + b_n \sin(nx) \right)
     \]
   - Any periodic function can be written as a sum of sines and cosines.

2. **Polynomial Basis (Curve Fitting & Regression)**  
   - Uses **polynomials** as basis functions:  
     \[
     f(x) = c_0 + c_1x + c_2x^2 + c_3x^3 + \dots
     \]
   - This is used in polynomial regression.

3. **Wavelets (Image & Audio Processing)**  
   - Uses **wavelet functions** like Haar or Daubechies wavelets for analyzing signals at multiple scales.

4. **Neural Networks (Machine Learning)**  
   - Activation functions (e.g., **ReLU, sigmoid, tanh**) act as basis functions to approximate complex functions.

### **Connection to Dot Product**  
- In **signal processing**, a function can be projected onto basis functions using the **dot product**, which measures how much of one function aligns with another.
- For example, **Fourier coefficients** are computed using dot products with sine and cosine functions.

