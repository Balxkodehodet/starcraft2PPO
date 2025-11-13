Kort forklart:

Denne agenten lærer å spille StarCraft II ved hjelp av en metode innen maskinlæring som kalles forsterkende læring (Reinforcement Learning, RL).
Agenten får poeng (belønninger) for gode handlinger og straff for dårlige, og prøver over tid å maksimere summen av belønninger.

Hvordan det fungerer

Agenten observerer spillet
Den får informasjon fra spillet – for eksempel antall arbeidere, hvor mange mineraler som er samlet, og hvor mange bygninger eller enheter som finnes.

Agenten gjør en handling
For eksempel: bygge arbeidere, samle mineraler, eller trene soldater.

Spillet svarer med en ny tilstand og en belønning
Hvis handlingen var bra (f.eks. flere ressurser eller en sterkere hær), får agenten positiv belønning.
Hvis handlingen var dårlig (f.eks. inaktive arbeidere eller tapte enheter), får den straff.

PPO-algoritmen oppdaterer strategien
PPO (Proximal Policy Optimization) er en stabil og effektiv RL-algoritme som justerer agentens atferd litt etter hver opplevelse — uten å endre for mye om gangen, for å unngå ustabil læring.

Over tid lærer agenten å bli bedre
Etter mange tusen episoder finner agenten gradvis strategier som maksimerer belønningen — for eksempel ved å balansere økonomi og militær styrke.

Self-play

I self-play spiller agenten mot en tidligere versjon av seg selv.
Dette gjør treningen mer realistisk og utfordrende — jo bedre agenten blir, desto bedre blir også motstanderen, slik at den fortsetter å lære nye strategier.

# StarCraft II PPO-agent med PySC2

Dette repositoriet inneholder et oppsett for forsterkende læring (Reinforcement Learning) ved bruk av **Proximal Policy Optimization (PPO)** for å trene en agent til å spille **StarCraft II** ved hjelp av **PySC2**-miljøet. Agenten bruker formede belønninger basert på spillets interne metrikker som mineraler, militære enheter og inaktive arbeidere for å lære effektive strategier.

---

## Funksjoner

- **PPO-agent** med CNN-policy for visuelle observasjoner.
- Agenten spiller mot en easy starcraft 2 bot i hoved filen som brukes (train_maskable_ppo_sc2.py)
- Masked actions sørger for at agenten velger valide handlinger av de 572 handlingene som er tilgjengelige.
- Formede belønninger som kombinerer:
  - Endring i poengsum
  - Mineraler samlet
  - Trenede arbeidere
  - Bruk av supply depot
  - Straff for inaktive arbeidere
- Støtte for self-play: agenten kan trene mot seg selv eller en fryst motspiller. (Ikke helt ferdig skrevet kode ennå)
- TensorBoard-logging for episoderesultater.
- Parallell trening med flere miljøer ved bruk av `SubprocVecEnv`.

---

## Avhengigheter

Bruk **Python 3.10**, da PySC2 krever denne versjonen.

### Python-pakker

- `pysc2`
- `gym` eller `gymnasium` <------ gymnasium blir brukt
- `stable-baselines3`
- `sb3-contrib` (for `MaskablePPO` dersom du bruker action masking)
- `torch`
- `numpy`
- `tensorboard`
- `matplotlib` (valgfritt, for visualiseringer)

Du kan installere avhengigheter med pip:

```bash
pip install pysc2 gym stable-baselines3 sb3-contrib torch numpy tensorboard matplotlib
```

StarCraft II-oppsett

Installer StarCraft II på maskinen din.

Sørg for at Maps-mappen er tilgjengelig (default er: C:\Program Files (x86)\StarCraft II\maps, lag maps mappen hvis ikke den er der allerede) og at PySC2 kan finne den (Den finner den automatisk hvis starcraft 2 er installert i default). Last ned maps (simple64 som agenten bruker i koden) fra blizzard sin github.

Anbefaling: Bruk en headless-versjon for raskere trening (visualize=False i SC2Env).

Bruk
Trening
python train_maskable_ppo_sc2.py <<-- Denne filen er hoved agenten, filen fungerer og de andre fungerer ikke optimalt ennå.

OPPDATERING: python v2_customCNN_ppo_10xdefault.py er den nyeste agenten med flere parametere da koden over produserte for få parametere og sakte trening etter flere millioner timesteps.. denne har 23.7 millioner parametere og tar opp flere 100MB plass å trene.
Den anbefales å bruke da den har oppdaterte rewards og bedre reward shaping.

Dette vil opprette flere parallelle miljøer og trene PPO-agenten mot en easy bot (ikke selv læring)

Modell-sjekkpunkter lagres periodisk via SaveEveryNStepsCallback.

TensorBoard-logger lagres i ./ppo_sc2_logs/. eller ./maskable_ppo_sc2_logs/ <<-- denne brukes akkurat nå

Evaluering / Self-Play

(selv læring agent ikke ferdig skrevet kode ennå)
Sett opponent_policy til en annen PPO-agent eller en skriptet policy for self-play trening. (i maskable_selfplay_ppo_sc2.py)

Miljøet støtter to agenter for self-play.

Belønningsstruktur

Formede belønninger kombinerer flere metrikker for å akselerere læring:

        shaped_reward = (
            0.5 * reward
            + 0.3 * delta_score
            + 0.15 * delta_army_power #reward for having supply used for army units
            + 0.05 * delta_vespene
            + 0.05 * delta_minerals
            + 0.05 * delta_workers  # reward for training new workers
            + 0.1 * delta_supply_despot
            - 0.2 * delta_idle_workers  # penalize idle workers
        )

Belønningene normaliseres for å unngå store hopp under PPO-oppdateringer.

Inaktive arbeidere straffes for å oppmuntre til effektiv ressursinnsamling.

Notater

Python 3.10 er nødvendig for kompatibilitet med PySC2.

Å kjøre flere miljøer (NUM_ENVS > 1) forbedrer treningstiden, men øker minnebruk.

Formede belønninger sørger for at agenten blir incentivert til å trene arbeidere, samle ressurser og bygge en effektiv hær.

- Tensorboard bruk:

  1.Åpne en terminal (cmd) eller den terminal du bruker
  2.Skriv i terminalen: ` tensorboard --logdir=MappenMedLogsHer`
  3.Åpne tensorboard i localhost, som oftest brukes port 6006
  4.Her kan du se mine logs eller se dine egne logs på treningen som er utført på ppo agenten.

Lisens

MIT-lisens. Du kan fritt modifisere eller utvide dette repositoriet for forskning eller egne prosjekter.

Forfatter

Utviklet av [balxkodehodet] & ChatGPT (ChatGPT skrev mesteparten av koden og jeg korrigerte eller forbedret koden ved å både skrive selv og spørre chatgpt mer detaljerte forespørsler)

Basert på implementasjoner fra PySC2 og Stable-Baselines3

```

```
