use num_traits::Float;
use std::ops;

/// A complex number stored as an in-phase / quadrature (I/Q) pair.
///
/// The type parameter `T` is the underlying scalar (typically `f32` for SDR
/// work, but any `Copy` type is accepted so that integer samples are also
/// supported).
///
/// # Fields
///
/// * `i` — the real (in-phase) component
/// * `q` — the imaginary (quadrature) component
///
/// # Examples
///
/// ```
/// use sdr::complex::Complex;
///
/// let c = Complex::new(1.0_f32, -0.5);
/// assert_eq!(c.i, 1.0);
/// assert_eq!(c.q, -0.5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Complex<T: Copy> {
    pub i: T,
    pub q: T,
}

impl<T: Copy> Complex<T> {
    /// Construct a new complex number from a real part `real` and imaginary
    /// part `imag`.
    #[inline(always)]
    pub const fn new(real: T, imag: T) -> Self {
        Self { i: real, q: imag }
    }
}

impl<T: Copy> From<(T, T)> for Complex<T> {
    /// Convert a `(real, imag)` tuple into a [`Complex`].
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl<T: Copy + ops::Add<Output = T>> ops::Add for Complex<T> {
    type Output = Self;

    /// Component-wise addition: `(a + b).i == a.i + b.i`, etc.
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            i: self.i + rhs.i,
            q: self.q + rhs.q,
        }
    }
}

impl<T: Copy + ops::Sub<Output = T>> ops::Sub for Complex<T> {
    type Output = Self;

    /// Component-wise subtraction: `(a - b).i == a.i - b.i`, etc.
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            i: self.i - rhs.i,
            q: self.q - rhs.q,
        }
    }
}

impl<T: Copy + ops::Mul<Output = T> + ops::Add<Output = T> + ops::Sub<Output = T>> ops::Mul
    for Complex<T>
{
    type Output = Self;

    /// Complex multiplication: `(a·b).i = a.i·b.i − a.q·b.q`,
    /// `(a·b).q = a.i·b.q + a.q·b.i`.
    fn mul(self, rhs: Self) -> Self::Output {
        let real = self.i * rhs.i - self.q * rhs.q;
        let imag = self.i * rhs.q + self.q * rhs.i;
        Self::new(real, imag)
    }
}

impl<T: Copy + PartialEq> PartialEq<(T, T)> for Complex<T> {
    /// Allow comparing a [`Complex`] directly against a `(real, imag)` tuple.
    fn eq(&self, other: &(T, T)) -> bool {
        self.i == other.0 && self.q == other.1
    }

    fn ne(&self, other: &(T, T)) -> bool {
        self.i != other.0 || self.q != other.1
    }
}

impl<T: Copy + ops::Neg<Output = T>> ops::Neg for Complex<T> {
    type Output = Self;

    /// Negate both components: `-c == Complex::new(-c.i, -c.q)`.
    fn neg(self) -> Self::Output {
        Self::new(-self.i, -self.q)
    }
}

impl<T: Copy + ops::Neg<Output = T>> Complex<T> {
    /// Return the complex conjugate: same real part, negated imaginary part.
    ///
    /// `c.conjugate() == Complex::new(c.i, -c.q)`
    pub fn conjugate(&self) -> Self {
        Self::new(self.i, -self.q)
    }
}

impl<T: Copy + ops::Mul<Output = T> + ops::Add<Output = T>> Complex<T> {
    /// Return the squared magnitude (power): `i² + q²`.
    ///
    /// No `sqrt` is performed, so this works for integer types too.
    pub fn norm_sq(&self) -> T {
        self.i * self.i + self.q * self.q
    }

    /// Multiply both components by a real scalar `s`.
    pub fn scale(&self, s: T) -> Self {
        Self::new(self.i * s, self.q * s)
    }
}

impl<T: Float + Copy> Complex<T> {
    /// Return the magnitude `√(i² + q²)`.
    pub fn abs(&self) -> T {
        (self.i * self.i + self.q * self.q).sqrt()
    }

    /// Return the phase angle `atan2(q, i)` in radians (`−π` to `+π`).
    pub fn arg(&self) -> T {
        self.q.atan2(self.i)
    }

    /// Construct from polar form: `r · (cos θ + j·sin θ)`.
    pub fn from_polar(r: T, theta: T) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn basic_ops() {
        let a: Complex<i32> = (1, -1).into();
        let b: Complex<i32> = (-1, 1).into();
        assert_eq!(a + b, (0, 0), "Addition");
        assert_eq!(a - b, (2, -2), "Subtraction");
        assert_eq!(a * b, (0, 2), "Multiplication");
    }

    #[test]
    fn norm_sq_integer() {
        let c: Complex<i32> = Complex::new(3, 4);
        assert_eq!(c.norm_sq(), 25);
    }

    #[test]
    fn norm_sq_float() {
        let c = Complex::new(3.0_f32, 4.0);
        assert!((c.norm_sq() - 25.0).abs() < 1e-6);
    }

    #[test]
    fn scale_both_components() {
        let c = Complex::new(2.0_f32, -3.0);
        let scaled = c.scale(4.0);
        assert!((scaled.i - 8.0).abs() < 1e-6);
        assert!((scaled.q - -12.0).abs() < 1e-6);
    }

    #[test]
    fn abs_345() {
        let c = Complex::new(3.0_f32, 4.0);
        assert!((c.abs() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn arg_angles() {
        assert!((Complex::new(1.0_f32, 0.0).arg() - 0.0).abs() < 1e-6);
        assert!((Complex::new(0.0_f32, 1.0).arg() - PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn from_polar_unit() {
        let c = Complex::<f32>::from_polar(1.0, 0.0);
        assert!((c.i - 1.0).abs() < 1e-6);
        assert!(c.q.abs() < 1e-6);
    }

    #[test]
    fn from_polar_roundtrip() {
        let orig = Complex::new(3.0_f32, 4.0);
        let r = orig.abs();
        let theta = orig.arg();
        let reconstructed = Complex::from_polar(r, theta);
        assert!((orig.i - reconstructed.i).abs() < 1e-5);
        assert!((orig.q - reconstructed.q).abs() < 1e-5);
    }
}
