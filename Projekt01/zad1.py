import math
from datetime import date

def biorhythm_value(days_lived, cycle_days):
    return math.sin((2 * math.pi / cycle_days) * days_lived)


def classify_and_message(name, label, value_today, value_tomorrow):
    if value_today > 0.5:
        print(f"{label}: {value_today:.3f}. Gratuluję dobry wynik, {name}! ")
    elif value_today < -0.5:
        print(f"{label}: {value_today:.3f}. Słaby dzień, {name}. Trzymaj się!")
        if value_tomorrow > value_today:
            print("Nie martw się. Jutro będzie lepiej!")
        elif value_tomorrow < value_today:
            print("Jutro może być trochę trudniej, zadbaj o odpoczynek.")
        else:
            print("Jutro będzie podobnie, spokojnie, dasz radę.")
    else:
        print(f"{label}: {value_today:.3f} Wynik neutralny, {name}.")


def main():
    name = input("Podaj imię: ").strip()
    year = int(input("Podaj rok urodzenia: ").strip())
    month = int(input("Podaj miesiąc urodzenia: ").strip())
    day = int(input("Podaj dzień urodzenia: ").strip())

    birth_date = date(year, month, day)
    today = date.today()

    days_lived = (today - birth_date).days
    if days_lived < 0:
        print("Uruchom program ponownie i wpisz poprawną datę.")
        return

    print(f"\nCześć, {name}!")
    print(f"Dzisiaj jest {today.isoformat()}.")
    print(f"To {days_lived}. dzień Twojego życia.\n")

    yp = biorhythm_value(days_lived, 23)   
    ye = biorhythm_value(days_lived, 28)   
    yi = biorhythm_value(days_lived, 33)  

    yp_t = biorhythm_value(days_lived + 1, 23)
    ye_t = biorhythm_value(days_lived + 1, 28)
    yi_t = biorhythm_value(days_lived + 1, 33)

    print("Twoje biorytmy na dziś:")
    classify_and_message(name, "Fizyczny", yp, yp_t)
    classify_and_message(name, "Emocjonalny", ye, ye_t)
    classify_and_message(name, "Intelektualny", yi, yi_t)


if name == "main":
    main()