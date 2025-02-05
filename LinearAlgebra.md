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

