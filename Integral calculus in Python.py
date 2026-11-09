import numpy as np
import sympy as sp

# Define symbolic variables
x = sp.Symbol('x')
u = sp.Symbol('u')
theta = sp.Symbol('theta', real=True)

print('=== 1. RIEMANN SUMS (Numerical Approximation) ===')
# Approximate the area under f(x) = x^2 from 0 to 2 using 1,000 midpoint rectangles
a, b, n = 0, 2, 1000
dx_num = (b - a) / n
midpoints = np.linspace(a + dx_num / 2, b - dx_num / 2, n)
riemann_sum = np.sum((midpoints**2) * dx_num)
print(f'Riemann Sum (n={n}): {riemann_sum:.6f} (Exact is {8/3:.6f})\n')

print('=== 2. BASIC ANTIDERIVATIVES & DEFINITE INTEGRALS ===')
f_basic = 3 * x**2 + sp.cos(x) + sp.exp(x)
indef_basic = sp.integrate(f_basic, x)
def_basic = sp.integrate(x**2, (x, 0, 2))
print('Indefinite Integral:', indef_basic, '+ C')
print('Definite Integral (0 to 2):', def_basic, '\n')

print('=== 3. U-SUBSTITUTION ===')
# Integral of 2x * cos(x^2) dx  (Let u = x^2 -> du = 2x dx)
u_expr = sp.cos(u)
u_integrated = sp.integrate(u_expr, u).subs(u, x**2)
print('U-Substitution Result:', u_integrated, '+ C\n')

print('=== 4. INTEGRATION BY PARTS ===')
# Formula: Integral(u dv) = u*v - Integral(v du)
# Example: Integral of x * e^x dx
u_part = x
dv_part = sp.exp(x)

du_part = sp.diff(u_part, x)
v_part = sp.integrate(dv_part, x)

ibp_result = u_part * v_part - sp.integrate(v_part * du_part, x)
print('Integration by Parts Result:', ibp_result, '+ C\n')

print('=== 5. PARTIAL FRACTION DECOMPOSITION ===')
# Example: Integral of (3x + 5) / (x^2 + 3x + 2) dx
rational_expr = (3 * x + 5) / (x**2 + 3 * x + 2)
decomposed = sp.apart(rational_expr, x)
pf_integral = sp.integrate(decomposed, x)
print('Decomposed Fractions:', decomposed)
print('Integrated Result:', pf_integral, '+ C\n')

print('=== 6. TRIGONOMETRIC SUBSTITUTION ===')
# Example: Integral of 1 / sqrt(16 - x^2) dx
trig_target = 1 / sp.sqrt(16 - x**2)

# Step A: Substitute x = 4*sin(theta), dx = 4*cos(theta) d(theta)
x_sub = 4 * sp.sin(theta)
dx_sub = sp.diff(x_sub, theta)
substituted = trig_target.subs(x, x_sub) * dx_sub

# Step B: Simplify sqrt(cos^2(theta)) -> cos(theta)
simplified_trig = sp.simplify(
    substituted.subs(sp.sqrt(sp.cos(theta) ** 2), sp.cos(theta))
)

# Step C: Integrate and back-substitute theta = arcsin(x/4)
theta_integral = sp.integrate(simplified_trig, theta)
final_trig_sub = theta_integral.subs(theta, sp.asin(x / 4))

print('Simplified Trig Form:', simplified_trig)
print('Final Answer in x:', final_trig_sub, '+ C\n')

print('=== 7. APPLICATIONS: AREA, VOLUME & ARC LENGTH ===')
# A. Area between y = x and y = x^2 from 0 to 1
area = sp.integrate(x - x**2, (x, 0, 1))

# B. Volume of revolution (Disk Method): y = sqrt(x) around x-axis from 0 to 4
volume_disk = sp.pi * sp.integrate(sp.sqrt(x) ** 2, (x, 0, 4))

# C. Arc length of y = x^(3/2) from 0 to 4
curve = x ** (sp.Rational(3, 2))
arc_length = sp.integrate(sp.sqrt(1 + sp.diff(curve, x) ** 2), (x, 0, 4))

print('Area Between Curves:', area)
print('Volume (Disk Method):', volume_disk)
print('Arc Length:', arc_length, '\n')

print('=== 8. IMPROPER INTEGRALS & TAYLOR SERIES ===')
# Improper integrals (limits to infinity)
improper_conv = sp.integrate(1 / x**2, (x, 1, sp.oo))
improper_div = sp.integrate(1 / x, (x, 1, sp.oo))

# Maclaurin series for sin(x) up to x^7
taylor_sin = sp.series(sp.sin(x), x, 0, 8)

print('Improper Integral (1/x^2, 1 to oo):', improper_conv)
print('Improper Integral (1/x, 1 to oo):', improper_div)
print('Maclaurin Series for sin(x):', taylor_sin)
