# Дообучение NanoVLM для MiniGrid

Компактный воспроизводимый проект для адаптации
[NanoVLM](https://github.com/huggingface/nanoVLM) к среде
[MiniGrid EmptyEnv](https://minigrid.farama.org/environments/minigrid/EmptyEnv/).

В проекте реализованы:

- supervised fine-tuning (SFT) на экспертных траекториях;
- GRPO с прямой генерацией действия;
- GRPO с кратким описанием состояния, дальнейшим планом и действием;
- сравнение базового prompt, prompt с описанием политики и балансировки действий.

Модель получает частичное RGB-наблюдение агента и выбирает одно из трёх
действий: `left`, `right` или `forward`.

## Структура проекта

```text
src/
  data/collect.py       сбор экспертных траекторий
  expert/policies.py    экспертные политики
  sft/train.py          SFT-обучение и оценка
  sft/plot.py           графики SFT
  grpo/train.py         батчированное GRPO-обучение
  grpo/evaluate.py      оценка checkpoint на фиксированных seed
  grpo/plot.py          графики GRPO
  config.py             prompts, действия и среды
  utils.py              общие функции NanoVLM и MiniGrid
patches/
  nanovlm-vqa-mask.patch
```

Датасеты, checkpoints, метрики и графики генерируются локально и не хранятся
в Git.

## Установка

Рекомендуются Python 3.10+ и GPU с поддержкой CUDA.

```bash
git clone https://github.com/1g0rp4vl/nanovlm_finetune.git nanovlm_finetune
cd nanovlm_finetune

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/huggingface/nanoVLM external/nanoVLM
git -C external/nanoVLM checkout 4e0c096
git -C external/nanoVLM apply ../../patches/nanovlm-vqa-mask.patch
```

Patch исправляет обработку результата токенизатора при построении loss mask
для ответа ассистента. Сам NanoVLM в этот репозиторий не включён.

Все дополнительные параметры и их значения по умолчанию доступны через
`--help`:

```bash
python -m src.data.collect --help
python -m src.sft.train --help
python -m src.grpo.train --help
python -m src.grpo.evaluate --help
```

## 1. Сбор экспертных данных

Основной эксперт движется по кратчайшему пути, когда зелёная цель видна, и
поворачивает направо, когда цель находится вне наблюдения. Такая стратегия
детерминирована и задаёт агенту последовательный способ поиска цели.

### Базовый датасет действий

```bash
python -m src.data.collect \
  --prompt action \
  --output-dir data/raw/action
```

### Датасет с описанием экспертной политики в prompt

```bash
python -m src.data.collect \
  --prompt policy \
  --output-dir data/raw/policy
```

### Датасет text + action

Target содержит два коротких предложения и заканчивается исполняемым
действием.

```bash
python -m src.data.collect \
  --prompt plan_action \
  --output-dir data/raw/plan_action
```

Каждый датасет содержит:

- `images/` — частичные RGB-наблюдения;
- `metadata.jsonl` — пути к изображениям, prompts и ответ эксперта;
- `episodes.csv` — return и результат каждого эпизода;
- `summary.json` — общую статистику и распределение действий.

Для эксперимента на нескольких размерах карты с экспертом, использующим только
частичное наблюдение:

```bash
python -m src.data.collect \
  --expert partial_observation \
  --mixed-envs \
  --episodes 5000 \
  --output-dir data/raw/partial_observation
```

## 2. SFT

Каждый запуск сохраняет checkpoints, `metrics.csv` и `final.json` внутри
указанной директории.

### Baseline: прямой вывод действия

```bash
python -m src.sft.train \
  --metadata data/raw/action/metadata.jsonl \
  --output-dir outputs/sft_action
```

### Улучшение 1: prompt с описанием политики

```bash
python -m src.sft.train \
  --metadata data/raw/policy/metadata.jsonl \
  --output-dir outputs/sft_policy
```

### Улучшение 2: мягкая балансировка действий

Частые действия прореживаются так, чтобы повысить долю `left` (редкое действие в датасете), примерно
сохранив соотношение между `right` и `forward`.

```bash
python -m src.sft.train \
  --metadata data/raw/policy/metadata.jsonl \
  --min-left-fraction 0.15 \
  --output-dir outputs/sft_policy_balanced
```

### SFT в формате text + action

```bash
python -m src.sft.train \
  --metadata data/raw/plan_action/metadata.jsonl \
  --output-dir outputs/sft_plan_action
```

Построение сравнительных графиков:

```bash
python -m src.sft.plot \
  --run "Action=outputs/sft_action/metrics.csv" \
  --run "Policy=outputs/sft_policy/metrics.csv" \
  --run "Balanced=outputs/sft_policy_balanced/metrics.csv" \
  --run "Plan+action=outputs/sft_plan_action/metrics.csv" \
  --output-dir reports/figures/sft
```

## 3. GRPO с прямым выводом действия

GRPO инициализируется SFT checkpoint. На каждом update используются четыре
стартовых состояния и четыре траектории для каждого состояния. Генерация,
выполнение траекторий и вычисление логарифмов вероятностей батчированы.

```bash
python -m src.grpo.train \
  --checkpoint outputs/sft_policy_balanced/final \
  --prompt policy \
  --output-dir outputs/grpo_action
```

Если SFT-модель уже достигает success rate, близкого к единице, у GRPO почти не
остаётся пространства для улучшения. Для более показательного эксперимента
можно использовать более ранний SFT checkpoint.

## 4. GRPO в формате text + action

```bash
python -m src.grpo.train \
  --checkpoint outputs/sft_plan_action/final \
  --prompt plan_action \
  --output-dir outputs/grpo_plan_action
```
 
Вероятность в Loss вычисляется по всей сгенерированной последовательности, а среда
исполняет действие, извлечённое из последнего слова ответа.

## 5. Оценка на фиксированных seed

Для честного сравнения SFT и GRPO следует оценивать checkpoints на одинаковых
200 seed:

```bash
python -m src.grpo.evaluate \
  --checkpoint outputs/sft_policy_balanced/final \
  --prompt policy \
  --output outputs/sft_policy_balanced/eval.json

python -m src.grpo.evaluate \
  --checkpoint outputs/grpo_action/final \
  --prompt policy \
  --output outputs/grpo_action/eval.json

python -m src.grpo.evaluate \
  --checkpoint outputs/grpo_plan_action/final \
  --prompt plan_action \
  --output outputs/grpo_plan_action/eval.json
```

Сохраняются success rate, средний return, средняя длина эпизода, доля truncated
эпизодов, доля некорректных действий и распределение действий.

## 6. Графики GRPO

```bash
python -m src.grpo.plot \
  --run "Action=outputs/grpo_action/metrics.jsonl" \
  --run "Plan+action=outputs/grpo_plan_action/metrics.jsonl" \
  --output-dir reports/figures/grpo
```

Для измерения sample efficiency номер update переводится в число использованных
эпизодов среды:

```text
episodes = update * rollout_batch_size * group_size
```

При параметрах по умолчанию один update использует `4 * 4 = 16` обучающих
эпизодов. Порог качества, например `success rate >= 0.8`, следует выбрать до
сравнения экспериментов.

## Разработка

```bash
pip install -r requirements-dev.txt
black --check src
ruff check src
```

## Отчет по экспериментам

Можно найти в файле report.pdf.
