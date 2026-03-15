use std::ops;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Complex<T: Copy> {
    pub i: T,
    pub q: T,
}

impl<T: Copy> Complex<T> {
    #[inline(always)]
    pub const fn new(real: T, imag: T) -> Self {
        Self { i: real, q: imag }
    }
}

impl<T: Copy> From<(T, T)> for Complex<T> {
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl<T: Copy + ops::Add<Output = T>> ops::Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            i: self.i + rhs.i,
            q: self.q + rhs.q,
        }
    }
}

impl<T: Copy + ops::Sub<Output = T>> ops::Sub for Complex<T> {
    type Output = Self;

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

    fn mul(self, rhs: Self) -> Self::Output {
        let real = self.i * rhs.i - self.q * rhs.q;
        let imag = self.i * rhs.q + self.q * rhs.i;
        Self::new(real, imag)
    }
}

impl<T: Copy + PartialEq> PartialEq<(T, T)> for Complex<T> {
    fn eq(&self, other: &(T, T)) -> bool {
        self.i == other.0 && self.q == other.1
    }

    fn ne(&self, other: &(T, T)) -> bool {
        self.i != other.0 || self.q != other.1
    }
}

impl<T: Copy + ops::Neg<Output = T>> ops::Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.i, -self.q)
    }
}

impl<T: Copy + ops::Neg<Output = T>> Complex<T> {
    pub fn conjugate(&self) -> Self {
        Self::new(self.i, -self.q)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic_ops() {
        let a: Complex<i32> = (1, -1).into();
        let b: Complex<i32> = (-1, 1).into();
        assert_eq!(a + b, (0, 0), "Addition");
        assert_eq!(a - b, (2, -2), "Subtraction");
        assert_eq!(a * b, (0, 2), "Multiplication");
    }
}
