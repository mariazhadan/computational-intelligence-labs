import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Stałe zadania
v0 = 50.0     
h = 100.0    
g = 9.81     

CEL_MIN, CEL_MAX = 50, 340
MARGINES = 5 

def zasieg(v0: float, h: float, alpha_deg: float, g: float = 9.81) -> float:
    """Zwraca zasięg d (w metrach) dla pocisku wystrzelonego z wysokości h."""
    alpha = math.radians(alpha_deg)
    vx = v0 * math.cos(alpha)
    vy = v0 * math.sin(alpha)

    # Jeśli kąt jest "dziwny" i vx ~ 0, to zasięg będzie ~0 (prawie pionowo)
    if abs(vx) < 1e-12:
        return 0.0

    t_flight = (vy + math.sqrt(vy * vy + 2.0 * g * h)) / g
    d = vx * t_flight
    return d

def rysuj_trajektorie(v0: float, h: float, alpha_deg: float, d: float, g: float = 9.81,
                      filename: str = "trajektoria.png") -> None:
    """Rysuje i zapisuje wykres trajektorii do pliku PNG."""
    alpha = math.radians(alpha_deg)

    # Punkty x od 0 do d
    x = np.linspace(0.0, d, 400)
    cos_a = math.cos(alpha)

    # Bezpiecznik (gdy cos ~ 0)
    if abs(cos_a) < 1e-12:
        x = np.array([0.0])
        y = np.array([h])
    else:
        y = h + x * math.tan(alpha) - (g * x**2) / (2.0 * (v0**2) * (cos_a**2))
        # drobna korekta numeryczna: nie schodź poniżej 0 na końcu
        y = np.maximum(y, 0.0)

    plt.figure()
    plt.plot(x, y)              # nie ustawiamy koloru ręcznie
    plt.grid(True)
    plt.xlabel("Odległość x [m]")
    plt.ylabel("Wysokość y [m]")
    plt.title("Trajektoria pocisku Warwolf")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    cel = random.randint(CEL_MIN, CEL_MAX)
    print(f"Cel znajduje się w odległości: {cel} m")
    print(f"Trafienie, jeśli pocisk spadnie w zakresie [{cel - MARGINES}, {cel + MARGINES}] m\n")

    proby = 0
    alpha_hit = None
    d_hit = None

    while True:
        proby += 1
        s = input(f"Próba #{proby} — podaj kąt α w stopniach (np. 35.5): ").strip()

        try:
            alpha = float(s)
        except ValueError:
            print("To nie jest liczba. Spróbuj ponownie.\n")
            continue

        if not (0.0 < alpha < 90.0):
            print("Podaj kąt w zakresie (0, 90) stopni.\n")
            continue

        d = zasieg(v0, h, alpha, g)
        print(f"Zasięg: {d:.2f} m")

        if (cel - MARGINES) <= d <= (cel + MARGINES):
            print("\nCel trafiony!")
            print(f"Liczba prób: {proby}")
            alpha_hit = alpha
            d_hit = d
            break
        else:
            if d < cel - MARGINES:
                print("Za krótko (zwiększ/zmień kąt).\n")
            else:
                print("Za daleko (zmniejsz/zmień kąt).\n")

    # Rysunek po trafieniu
    rysuj_trajektorie(v0, h, alpha_hit, d_hit, g, filename="trajektoria.png")
    print("\nZapisano wykres: trajektoria.png")

if __name__ == "__main__":
    main()
