# 500题三赛道JSON接口

## 分配方案

- 赛道一 `track_1`：200题（40%）
- 赛道二 `track_2`：150题（30%）
- 赛道三 `track_3`：150题（30%）
- 合计：500题

比例由项目配置管理；如老师调整比例，应同步修改模板、Schema、校验程序和提示词配置。

全部500题都进入纯LLM与RAG的对照流程，每题由裁判模型评判一次。`track`用于选择相应的赛道提示词与评价语境。

## 单题字段

```json
{
  "id": "Q001",
  "question": "问题正文",
  "track": "track_1",
  "expected_evidence_type": "guideline",
  "notes": "判分要点、问题边界或出题说明",
  "should_abstain": false
}
```

字段约定：

- `id`：从 `Q001` 到 `Q500`，不得重复或跳号。
- `question`：正式问题正文。
- `track`：只能是 `track_1`、`track_2`、`track_3`。
- `expected_evidence_type`：只能是 `guideline`、`systematic_review`、`RCT`、`observational`、`mixed`、`other`、`insufficient`。
- `notes`：判分要点、适用边界或出题说明；没有时填空字符串。
- `should_abstain`：该题是否预期模型拒答或明确说明证据不足。

论文来源和证据内容不写进题目对象。系统在运行时根据问题检索，得到 `list[Metadata]` 后分别传给RAG回答模块和裁判核验模块。

## 文件用途

- `questions_500.template.json`：正式题库的空白容器；收到500题后向 `questions` 数组填入记录，并把 `status` 改成 `draft` 或 `frozen`。
- `question_dataset.schema.json`：供其他同学或程序校验JSON结构。
- `question.example.json`：单题填写示例，不属于正式题库。
- `validate_questions.py`：检查总数、编号、字段、枚举值和三赛道数量。

## 冻结规则

正式评测前应将测试集版本固定为 `frozen`。模型运行期间不得为了提高胜率临时删除、替换或改写不利题目。
