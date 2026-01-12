from math import sqrt

LOW = 0.01
UPPER = 0.99
TICK = 0.01

# Choose ONE:
# k = 10000.0                 # normalized curve
k = (200.0 / (sqrt(UPPER) - sqrt(LOW)))**2   # if you want total quote Δy across range = 100

sqrt_k = sqrt(k)

p = LOW
while p + TICK <= UPPER + 1e-12:
    p_next = round(p + TICK, 2)

    # size in token A for moving along x*y=k from p -> p_next
    size = sqrt_k * (1.0 / sqrt(p) - 1.0 / sqrt(p_next))

    print(f"{p:.2f} {size:.8f} {size * p:.8f} ")

    p = p_next
