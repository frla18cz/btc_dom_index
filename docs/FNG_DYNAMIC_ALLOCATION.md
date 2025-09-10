# FNG Dynamic Allocation – Uživatelský průvodce

Tento průvodce vysvětluje, jak v aplikaci navrhovat dynamické váhy BTC/ALT podle Fear & Greed Indexu (FNG). Nový Quick Designer umožňuje rychlé, intuitivní nastavení s vizuálním náhledem a volitelným pokročilým editorem tabulky.

## Co je FNG dynamická alokace
- Cíl: Měnit váhy BTC (`BTC_w`) a ALT (`ALT_w`) dle úrovně FNG (0 = extrémní strach, 100 = extrémní chamtivost).
- Implementace: Backtester přiřazuje pro každý týden FNG do 10bodových binů (0, 10, …, 90) a použije váhy z mapy `{bin → (BTC_w, ALT_w)}`.

## Kde to najdete v UI
- Sidebar → sekce „Strategy Configuration“ → zaškrtnout „Enable FNG‑based Dynamic Allocation“.
- Nový blok „FNG Dynamic Allocation“:
  - „Use Quick Designer (recommended)“ – zapne návrhář s náhledem.
  - Expander „Advanced table editor“ – ruční doladění hodnot v tabulce.

## Quick Designer – principy
- Režim návrhu:
  - Lock total leverage: drží součet `BTC_w + ALT_w` konstantní napříč biony.
  - Independent weights: `BTC_w` a `ALT_w` nastavujete nezávisle (součet se může měnit).
- Endpoints: nastavíte hodnoty „na začátku“ (FNG=0) a „na konci“ (FNG=90) pro BTC a/nebo ALT.
- Curve shape: tvar interpolace mezi endpointy – Linear, Ease‑in, Ease‑out, S‑curve.
- Náhled: malý line chart v sidebaru ukazuje průběh `BTC_w` a `ALT_w` přes biny 0..90.

### Curve Shapes – vysvětlení a kdy je použít
- Linear: přímé, rovnoměrné přechody (vzorec: f(t)=t). Vhodné jako výchozí volba.
- Ease‑in: pomalý začátek, rychlejší konec (f(t)=t²). Víc konzervativní v nízkém FNG, agresivnější ve vysokém.
- Ease‑out: rychlý začátek, pozvolné dojezdy (f(t)=1−(1−t)²). Rychle reaguje při nízkém FNG, stabilizuje se ve vysokém.
- S‑curve: plynulé „S“ (smoothstep f(t)=t²·(3−2t)). Menší citlivost u krajů, jemnější střed.

Tip: t je normalizovaná hodnota v rozmezí 0..1 od FNG=0 do FNG=90. `BTC_w = BTC_low + (BTC_high−BTC_low)·f(t)`, obdobně pro `ALT_w`.

## Postup krok za krokem
1. V sidebaru zapněte „Enable FNG‑based Dynamic Allocation“.
2. Zapněte „Use Quick Designer (recommended)“.
3. Zvolte „Design mode“:
   - Lock total leverage a vyberte „Total leverage“ (např. 2.50×).
   - Nebo Independent weights.
4. Nastavte endpoints:
   - BTC at FNG=0 a BTC at FNG=90.
   - ALT at FNG=0 a ALT at FNG=90 (u lock režimu se vypočte automaticky z total leverage).
5. Zvolte „Curve shape“ (Linear je nejpřehlednější výchozí volba).
6. Zkontrolujte náhled křivek; otevřete „Advanced table editor“ pro jemné úpravy konkrétních binů.
7. Spusťte backtest – mapování se propíše do `fng_weight_bins` v analyzéru.

## Příklady nastavení
- Defenzivní (Lock total leverage):
  - Total leverage: 2.0×
  - BTC at FNG=0: 1.75× → ALT=0.25×
  - BTC at FNG=90: 1.25× → ALT=0.75×
  - Shape: S‑curve (plynulejší střed)
- Agresivní (Independent weights):
  - BTC at FNG=0: 1.5×, BTC at FNG=90: 1.0×
  - ALT at FNG=0: 0.5×, ALT at FNG=90: 1.5×
  - Shape: Ease‑out (rychlejší nárůst při vyšším FNG)

## Validace a omezení
- Celková páka: UI zobrazuje „Total leverage“. Doporučení ≤ 3.0×.
- Výsledná tabulka: hodnoty jsou v násobcích (×). Např. 1.75 = 175 % kapitálu.
- Pokud používáte „tuesday_lookahead“ FNG politiku, vědomě tím porušujete kauzalitu (FNG z úterý přiřazen pondělí) – UI na to upozorňuje.

## Jak to funguje uvnitř
- `app.py`:
  - Quick Designer vytváří DataFrame s řádky pro biny 0..90 a sloupci `BTC_w`, `ALT_w`.
  - Volitelný `st.data_editor` umožní ruční úpravy hodnot.
  - Při backtestu se tabulka převede na mapu `{int(bin_start): {"btc_w": float, "alt_w": float}}` a předá `backtest_rank_altbtc_short(..., fng_weight_bins=...)`.
- `analyzer_weekly.py`:
  - Podle týdenního FNG vybere správný bin a použije odpovídající váhy.

## Tipy
- Začněte s „Lock total leverage“ (snadná kontrola celkového rizika), potom přepněte na „Independent weights“, pokud chcete jinou páku BTC/ALT podle FNG.
- „S‑curve“ dává plynulý průběh bez ostrých hran v polovině rozsahu.
- Pokud chcete „kapsy“ preferencí, použijte Quick Designer pro tvar a konkrétní bity doladíte v „Advanced table editor“.

## Běžné potíže
- „No data available…“ – zvolte období v rámci dostupných dat v sidebaru, nebo aktualizujte data (Update Data).
- „Total leverage exceeds 3.0“ – snižte koncové hodnoty nebo zvolte menší total leverage.

---
Pro zpětnou vazbu k návrháři nebo pro předpřipravené presety (Defenzivní/Neutrální/Agresivní) otevřete issue nebo PR.
