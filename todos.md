1. 加一个多个结果一起跑然后相关性分析的代码
![](res/2026-01-06-13-15-52.png)
2. 加一个多个结果一起跑然后将结果整理到一张表中的代码
![](res/2026-01-06-13-11-49.png)
3. 后续的仓位管理、风险管理、测试等：https://claude.ai/chat/7e1839ab-abdf-4d05-be64-304d08362d2f
4. 研究其他期货：https://claude.ai/chat/12cf948f-1c4c-4334-8c34-efa52c6f6d79
5. 还是量化为主，基本面作为否定量化的终极办法，不要一开始先陷入基本面的一些所谓经验中
6. https://claude.ai/chat/7e1839ab-abdf-4d05-be64-304d08362d2f
7. ![仓位控制开发思路](image.png)
8. 跑代码的步骤：PORTFOLIO_WORKFLOW_GUIDE.md
9. ![其他核心关注问题](image-1.png)
10. 纯粹的价量因子是无法预测长期走势的：nq_trading_system_research_roadmap

python walk_forward_validator.py --data-start 20200101 --data-end 20241231--timeframe d1 --train-months 24 --test-months 12 --step-months 12 --top-n 20



# 整体流程步骤
1. 处理数据
2. 跑单独策略，用bt_main或者run_all_strategies_2020_2024
3. 跑相关性，看热力图低相关性
4. 跑策略组合，用低相关性做组合