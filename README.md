## Szoftver Telepítése és Indítása

1. **Telepítse a szoftvert** a megadott telepítési fájlt használva.
2. **Indítsa el a backend szolgáltatást**.

---

## Tesztelési Útmutató

A tesztelés során **használja a mappában található képfájlt** bemenetként.

---

## API Backend Konténer Indítása

A backend futtatásához kövesse az alábbi lépéseket egy Docker-kompatibilis környezetben:

Húzza le az API backend konténer képét a Docker Hub-ról majd futtassa.

docker pull karesz/aiapi:v1.0.0

docker run -d -p 5000:5000 --gpus=all karesz/aiapi:v1.0.0
