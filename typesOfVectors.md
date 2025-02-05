

### **1️⃣ Zero Vector (\(\mathbf{0}\))**
- A vector with **zero magnitude**:  
  \[
  \mathbf{0} = (0, 0) \text{ in 2D or } (0, 0, 0) \text{ in 3D}
  \]
- It has **no specific direction**.
- Used as an **identity element** in vector addition.

---

### **2️⃣ Unit Vector (\(\mathbf{\hat{v}}\))**
- A vector with **magnitude 1**.  
- Used to **represent direction only**, not length.  
- Found using:
  \[
  \mathbf{\hat{v}} = \frac{\mathbf{v}}{|\mathbf{v}|}
  \]
- Example: Standard unit vectors in 3D:
  \[
  \mathbf{\hat{i}} = (1,0,0), \quad \mathbf{\hat{j}} = (0,1,0), \quad \mathbf{\hat{k}} = (0,0,1)
  \]

---

### **3️⃣ Position Vector (\(\mathbf{r}\))**
- Represents a point's position relative to the origin.  
- Example: \( \mathbf{r} = (x, y, z) \) gives coordinates of a point in space.  
- Used in **physics and computer graphics** to describe object positions.

---

### **4️⃣ Direction Vector (\(\mathbf{d}\))**
- Defines the **direction of a line** or movement.  
- Example: The line \( y = 2x \) has direction vector:
  \[
  \mathbf{d} = (1, 2)
  \]
- Used in **parametric equations of lines**.

---

### **5️⃣ Normal Vector (\(\mathbf{N}\))**
- A vector **perpendicular** to a surface or a line.  
- For a line \( Ax + By + C = 0 \), the normal vector is:
  \[
  \mathbf{N} = (A, B)
  \]
- Used in **plane equations** and **graphics rendering**.

---

### **6️⃣ Tangent Vector (\(\mathbf{T}\))**
- A vector **tangent to a curve** at a specific point.  
- Used in **differential geometry and physics**.

---

### **7️⃣ Gradient Vector (\(\nabla f\))**
- Used in calculus to show **rate of change** of a function.  
- Example:
  \[
  \nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
  \]
- Points in the **direction of greatest increase** of \( f(x,y) \).

---

### **8️⃣ Basis Vectors**
- **A set of vectors that can represent any vector in a space**.
- Example: Standard basis vectors in 3D:
  \[
  \mathbf{e}_1 = (1,0,0), \quad \mathbf{e}_2 = (0,1,0), \quad \mathbf{e}_3 = (0,0,1)
  \]

---

### **9️⃣ Projection Vector**
- The vector **component of one vector along another**.
- Formula:
  \[
  \text{Proj}_{\mathbf{B}} \mathbf{A} = \left( \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{B}|^2} \right) \mathbf{B}
  \]
- Used in **physics (work done by a force), signal processing, and 3D graphics**.

---

### **🔟 Eigenvector**
- A special vector that only **scales** when a matrix transformation is applied.
- Used in **machine learning, quantum mechanics, and linear algebra**.

---

