# OLMo Zero-Order Fork

Этот репозиторий - форк `allenai/OLMo` с добавленными zero-order оптимизаторами и готовыми конфигами для экспериментов на wiki-данных.

Основные отличия от апстрима:
- добавлены ZO-оптимизаторы (включая `lozo`, `zo_adam`, `ldsd_muon`, `ldsd_rl`, `ldsd_rl_adamm`, `ldsd_rl_sgd`);
- добавлены рабочие и smoke-конфиги в `configs/wiki/` и `configs/smoke/`;
- фокус на практичном запуске обучения через `scripts/train.py`.

---

## Быстрый старт

### 1) Установка

```bash
git clone <your-fork-url>
cd OLMo-Zero-Order-Optimization
pip install -e .[all]
```

Если используете `uv`, можно запускать команды через `uv run ...`.

### 2) Проверка установки

```bash
python -c "from olmo import check_install; check_install(cuda=True)"
```

### 3) Smoke-запуск (рекомендуется первым делом)

```bash
uv run torchrun --nproc_per_node=1 scripts/train.py \
  configs/smoke/OLMo2-wiki-1gpu-zo-adam-smoke.yaml
```

Если smoke проходит, можно переходить к длинным экспериментам из `configs/wiki/`.

---

## Как запускать обучение

Общий шаблон:

```bash
torchrun --nproc_per_node=<NUM_GPUS> scripts/train.py <PATH_TO_CONFIG>
```

Пример:

```bash
uv run torchrun --nproc_per_node=2 scripts/train.py \
  configs/wiki/OLMo2-1B-wiki-ddp-zo-adam_new.yaml
```

Переопределение параметров из CLI:

```bash
uv run torchrun --nproc_per_node=2 scripts/train.py \
  configs/wiki/OLMo2-1B-wiki-ddp-zo-adam_new.yaml \
  --optimizer.learning_rate=8e-5 \
  --global_train_batch_size=256
```

---

## Конфиги: что где лежит

### `configs/wiki/` - основные long-run конфиги (ключевая папка)

Здесь находятся полноразмерные эксперименты на wiki-данных.

На что смотреть в первую очередь:
- `configs/wiki/OLMo2-1B-wiki-ddp-zo-adam_new.yaml` - обновленный DDP-конфиг для `zo_adam`;
- `configs/wiki/OLMo2-1B-wiki-ddp-zo-adam.yaml` - базовый DDP-конфиг для `zo_adam`;
- `configs/wiki/OLMo2-1B-wiki-1gpu-lozo.yaml` - single-GPU вариант с `lozo`;
- `configs/wiki/OLMo2-1B-wiki-ddp-ldsd-rl-sgd.yaml` - RL-SGD вариант (`ldsd_rl_sgd`);
- `configs/wiki/OLMo2-1B-wiki-ddp-ldsd-rl-adamm.yaml` - RL-AdaMM вариант (`ldsd_rl_adamm`);
- `configs/wiki/OLMo2-1B-wiki-ddp-ldsd-muon.yaml` - вариант с `ldsd_muon`.

### `configs/smoke/` - короткие проверки пайплайна

Мини-модели и короткая длительность, чтобы быстро проверить:
- корректность окружения;
- загрузку данных;
- работу конкретного оптимизатора.

Примеры:
- `configs/smoke/OLMo2-wiki-1gpu-zo-adam-smoke.yaml`
- `configs/smoke/OLMo2-wiki-1gpu-lozo-smoke.yaml`
- `configs/smoke/OLMo2-ddp-ldsd-rl-sgd-smoke.yaml`

### Прочие папки в `configs/`

- `official-*`, `annealing/`, `microannealing/`, `tiny/` - конфиги из оригинального OLMo/связанных экспериментов;
- для ZO-экспериментов в этом форке обычно достаточно начать с `wiki/` и `smoke/`.

---

## Данные и пути

В конфигах встречаются два типа путей:
- HTTP URL (стриминг данных);
- локальные `.npy` пути (например, `/home/.../data/...`).

Для ZO/RL-скриптов в этом форке используйте только локальные данные из папки `data/` этого репозитория.

Перед любым запуском сначала скачайте LFS-объекты:

```bash
git lfs pull
```

Перед запуском обязательно переопределите секции `data.paths` и `evaluators[].data.paths` в выбранном конфиге так, чтобы они ссылались на вашу локальную папку `data/`.

Пример:

```bash
uv run torchrun --nproc_per_node=2 scripts/train.py \
  configs/wiki/OLMo2-1B-wiki-ddp-zo-adam_new.yaml \
  --data.paths='["<repo_path>/data/train/0_00000.npy"]' \
  --evaluators.0.data.paths='["<repo_path>/data/validation/0_00000.npy"]'
```

Пример, когда пути переопределяются прямо в YAML-конфиге:

```yaml
# configs/wiki/OLMo2-1B-wiki-ddp-zo-adam_new.yaml
evaluators:
  - label: dolma_wiki-validation
    type: lm
    data:
      paths:
        - <repo_path>/data/validation/0_00000.npy

data:
  paths:
    - <repo_path>/data/train/0_00000.npy
    - <repo_path>/data/train/0_00001.npy
```

Если нужны локальные данные для wiki-экспериментов, используйте папку `data/` (локальное размещение датасетов).

---

## Практические замечания по ZO-обучению

- `zo_adam` и RL-варианты обычно требуют больше времени на шаг, чем первый порядок.
- DDP-конфиги должны запускаться с `distributed_strategy: ddp`.
- Проверяйте совместимость батчей:
  - `global_train_batch_size` должен корректно делиться на число процессов;
  - `device_train_microbatch_size` должен укладываться в память GPU.
- При OOM сначала уменьшайте `device_train_microbatch_size`, затем при необходимости `global_train_batch_size`.

---

## Где смотреть реализацию оптимизаторов

- `olmo/optim.py` - сборка и подключение оптимизаторов в тренере;
- `olmo/ldsd_optim.py` - реализации LDSD/RL-семейства;
- `olmo/config.py` - типы и поля конфигурации оптимизаторов.

---

## Полезные команды

Запуск обычного (first-order) baseline:

```bash
uv run torchrun --nproc_per_node=1 scripts/train.py \
  configs/wiki/OLMo2-1B-wiki-1gpu.yaml
```

Запуск LoZO baseline:

```bash
uv run torchrun --nproc_per_node=1 scripts/train.py \
  configs/wiki/OLMo2-1B-wiki-1gpu-lozo.yaml
```

Запуск ZoAdam (DDP):

```bash
uv run torchrun --nproc_per_node=2 scripts/train.py \
  configs/wiki/OLMo2-1B-wiki-ddp-zo-adam_new.yaml
```

---

## Логи и артефакты

- `save_folder` в конфиге определяет, куда сохраняются чекпоинты;
- метрики/логи идут в консоль и (при настройке) в Weights & Biases.

---

## Апстрим OLMo

Если нужна оригинальная документация/релизы OLMo, смотрите апстрим:
- [allenai/OLMo](https://github.com/allenai/OLMo)
- [allenai/OLMo-core](https://github.com/allenai/OLMo-core)
