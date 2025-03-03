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



The **dot product** (also called the **scalar product**) of vectors has many practical applications across different fields. Here are some key applications:

 **1. Physics & Engineering**  
- **Work Done by a Force**:  
  \[
  W = \mathbf{F} \cdot \mathbf{d} = |\mathbf{F}| |\mathbf{d}| \cos \theta
  \]
  - Used to calculate the work done when a force is applied along a displacement.

- **Electric and Magnetic Fields**:  
  - Used in computing power in AC circuits: \( P = V \cdot I \cos \theta \).
  - Helps in determining how much of a force or field acts in a given direction.

 **2. Computer Science & Machine Learning**  
- **Cosine Similarity in NLP & Data Science**:  
  \[
  \cos \theta = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}| |\mathbf{B}|}
  \]
  - Measures similarity between text documents or feature vectors in recommendation systems.

- **3D Graphics & Computer Vision**:  
  - Used in lighting calculations (Phong reflection model).
  - Determines angles between surfaces for shading and rendering.

 **3. Robotics & Mechanics**  
- **Joint Angles in Kinematics**:  
  - Helps in computing angles between robotic arms.
  
- **Torque & Rotation**:  
  - Used in determining components of force along a particular direction.

 **4. Geometry & Trigonometry**  
- **Projection of One Vector onto Another**:  
  - Used to find how much of one vector lies along another.
  
- **Angle Between Two Vectors**:  
  - Essential in navigation, physics, and structural analysis.

 **5. Economics & Finance**  
- **Portfolio Optimization**:  
  - Measures correlation between different financial assets.

- **Linear Regression**:  
  - Used in least squares fitting, which is fundamental to statistical modeling.

 **6. Signal Processing**  
- **Fourier Transforms & Filtering**:  
  - Helps in analyzing signals by projecting them onto basis functions.


### **1. Scalar Projection (Component of One Vector onto Another)**  
The **scalar projection** of a vector **\( \mathbf{A} \)** onto another vector **\( \mathbf{B} \)** is the length (magnitude) of the component of **\( \mathbf{A} \)** in the direction of **\( \mathbf{B} \)**.  

#### **Formula:**
\[
\text{Scalar Projection of } \mathbf{A} \text{ on } \mathbf{B} = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{B}|}
\]
This gives a **real number** (scalar) representing how much **\( \mathbf{A} \)** lies along **\( \mathbf{B} \)**.  

#### **Example:**  
If **\( \mathbf{A} = (3, 4) \)** and **\( \mathbf{B} = (1, 2) \)**, then:

1. Compute the **dot product**:  
   \[
   \mathbf{A} \cdot \mathbf{B} = (3 \times 1) + (4 \times 2) = 3 + 8 = 11
   \]
2. Find **\( |\mathbf{B}| \)** (magnitude of **\( \mathbf{B} \)**):  
   \[
   |\mathbf{B}| = \sqrt{1^2 + 2^2} = \sqrt{5}
   \]
3. Compute **scalar projection**:  
   \[
   \frac{11}{\sqrt{5}}
   \]

---

### **2. Vector Projection (Projection Vector of \( \mathbf{A} \) onto \( \mathbf{B} \))**  
The **vector projection** gives the actual vector component of **\( \mathbf{A} \)** in the direction of **\( \mathbf{B} \)**.

#### **Formula:**
\[
\text{Vector Projection of } \mathbf{A} \text{ on } \mathbf{B} = \left( \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{B}|^2} \right) \mathbf{B}
\]
This formula provides a vector in the same direction as **\( \mathbf{B} \)**.

#### **Example:**  
Using the same **\( \mathbf{A} = (3, 4) \)** and **\( \mathbf{B} = (1, 2) \)**:

1. Compute **\( |\mathbf{B}|^2 \)**:  
   \[
   |\mathbf{B}|^2 = 1^2 + 2^2 = 5
   \]
2. Compute **scalar factor**:  
   \[
   \frac{11}{5} = 2.2
   \]
3. Multiply by **\( \mathbf{B} \)**:  
   \[
   \mathbf{P} = 2.2 \times (1, 2) = (2.2, 4.4)
   \]

---

### **Key Differences:**
| Type | Formula | Output |
|------|---------|--------|
| **Scalar Projection** | \( \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{B}|} \) | A number (magnitude only) |
| **Vector Projection** | \( \left( \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{B}|^2} \right) \mathbf{B} \) | A vector (direction + magnitude) |



**Basis Function**
A basis function is a fundamental building block used to represent more complex functions in terms of simpler, well-defined components. It comes up in linear algebra, signal processing, machine learning, and numerical methods.
A **basis function** is a fundamental building block used to represent more complex functions in terms of simpler, well-defined components. It comes up in **linear algebra, signal processing, machine learning, and numerical methods**.  

### **Why is it Important?**  
In many fields, we approximate or decompose a function using a **linear combination** of basis functions. These basis functions are usually chosen to make computations easier.

 **Examples of Basis Functions**  

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

 **Connection to Dot Product**  
- In **signal processing**, a function can be projected onto basis functions using the **dot product**, which measures how much of one function aligns with another.
- For example, **Fourier coefficients** are computed using dot products with sine and cosine functions.

![image](https://github.com/user-attachments/assets/c253d09f-b418-4601-82f6-671b4c5fa4be)

Let’s break these concepts down in a simple way:  

---

## **1. Eigenvalues and Eigenvectors**
### **What are they?**
- **Eigenvalues** and **eigenvectors** are special properties of square matrices that reveal how the matrix transforms space.
- If we multiply a matrix \( A \) by a vector \( v \), and the result is just a **scaled** version of \( v \), then \( v \) is an **eigenvector**, and the scaling factor is its **eigenvalue** (\( \lambda \)).

### **Mathematical Definition**
For a square matrix \( A \):
\[
A v = \lambda v
\]
where:
- \( v \) is the **eigenvector** (a nonzero vector)
- \( \lambda \) is the **eigenvalue** (a scalar)

### **Why are they important?**
- Used in **machine learning** (e.g., PCA – Principal Component Analysis)
- Helps in **stability analysis** of systems
- Used in **Google’s PageRank algorithm**  

### **Example**
If  
\[
A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}
\]
has an eigenvector \( v = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \),  
then \( A v = \lambda v \), meaning it just gets scaled.

To compute **eigenvalues**, solve:
\[
\det(A - \lambda I) = 0
\]
which gives the values of \( \lambda \).

---

## **2. Singular Value Decomposition (SVD)**
### **What is it?**
- SVD is a method of **decomposing** any (even non-square) matrix into three **special matrices**:
  \[
  A = U \Sigma V^T
  \]
  where:
  - \( U \) = Left **orthogonal** matrix (columns are eigenvectors of \( A A^T \))
  - \( \Sigma \) = **Diagonal matrix** of singular values (like eigenvalues but always positive)
  - \( V^T \) = Right **orthogonal** matrix (columns are eigenvectors of \( A^T A \))

### **Why is it useful?**
- **Dimensionality reduction** (PCA uses SVD)
- **Image compression** (JPEG uses it)
- **Solving ill-conditioned systems** (numerically stable than normal inverse)

### **Example**
If  
\[
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
\]
then:
- \( U \) and \( V \) are orthogonal
- \( \Sigma \) contains the **singular values**.

In Python, compute it using:
```python
U, S, VT = np.linalg.svd(A)
```

---

## **3. Norm of a Matrix**
### **What is it?**
- The **norm** of a matrix is a single number that represents the matrix’s **size** or **magnitude**.

### **Common Norms**
1. **Frobenius Norm**: 
   \[
   ||A||_F = \sqrt{\sum A_{ij}^2}
   \]
   (Sum of squares of all elements)
   
2. **L1 Norm** (Max column sum):  
   \[
   ||A||_1 = \max \sum |A_{ij}|
   \]

3. **L∞ Norm** (Max row sum):  
   \[
   ||A||_{\infty} = \max \sum |A_{ij}|
   \]

### **Why is it useful?**
- Measures the **size of transformation** a matrix performs.
- Used in **optimization** and **error analysis**.

### **Example in NumPy**
```python
np.linalg.norm(A, 'fro')  # Frobenius norm
```

---

## **4. QR Decomposition**
### **What is it?**
- QR decomposition **factorizes** a matrix into:
  \[
  A = QR
  \]
  where:
  - \( Q \) = Orthogonal matrix (\( Q^T Q = I \))
  - \( R \) = Upper triangular matrix

### **Why is it useful?**
- Used in **solving linear systems** and **Least Squares Regression**.
- Basis for many **numerical algorithms**.

### **Example in NumPy**
```python
Q, R = np.linalg.qr(A)
```

---

### **Final Summary**
| Concept | Meaning | Application |
|---------|---------|-------------|
| **Eigenvalues & Eigenvectors** | Special vectors where matrix multiplication only scales them | PCA, stability analysis, Google PageRank |
| **Singular Value Decomposition (SVD)** | Breaks matrix into orthogonal components | Dimensionality reduction, image compression |
| **Matrix Norm** | Measures size of a matrix | Error estimation, optimization |
| **QR Decomposition** | Factorizes matrix into orthogonal & upper triangular form | Solving equations, least squares |

Let me know if you need more details! 🚀s
