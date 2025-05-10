
import asyncio

# from runLLM import TaskPlanning
from src.run import TaskPlanning
async def test_task_planning():
    # 测试用例1.2：带VLM+ LLM的情况
    planner = TaskPlanning(instruction="分析当前任务完成情况，生成决策")
    result = await planner.__forward__(
        "data/input/episode_000000-high.mp4",
        "data/input/episode_000000-left.mp4",
        "data/input/episode_000000-right.mp4"
    )
    print("最终决策:", result.get("last_decision"))


    # 测试用例1.1：带指令的情况
    # planner_with_instruction = TaskPlanning(instruction="执行装配检测任务,分析下一步应该怎么做？")
    # results = planner_with_instruction.__forward__(
    #     "data/episode_000000-high.mp4",
    #     "data/episode_000000-left.mp4",
    #     "data/episode_000000-right.mp4"
    # )
    # # print(result1)
    # # 输出结果
    # for i, result in enumerate(results):
    #     print(f"帧组 {i + 1} 决策:")
    #     print(f"推理：{result['decision']['reasoning']}")
    #     print(f"动作：{result['decision']['action']}")
    #     print("-" * 50)



    # #测试用例1.2：不带指令的情况
    # planner_without_instruction = TaskPlanning()
    # result2 = planner_without_instruction.__forward__(
    #     "data/sample1.png",
    #     "data/sample2.png",
    #     "data/sample3.png"
    # )
    # print("Test Case 1.2 (without instruction):", result2)

    # 测试用例2：不带指令的情况
    # planner_without_instruction = TaskPlanning(instruction="分析和比较这三张图片")
    # result2 = planner_without_instruction.__forward1__(
    #     "https://aigc-files.bigmodel.cn/api/cogview/202411181621355e276522adac433a_0.png",
    #     "https://aigc-files.bigmodel.cn/api/cogview/2024111816214822f9ee58eefa48bc_0.png",
    #    "https://aigc-files.bigmodel.cn/api/cogview/2024111816415481bfd2e6ef99445e_0.png"
    # )
    # print("Test Case 2 (without instruction):", result2)


if __name__ == "__main__":
    asyncio.run(test_task_planning())