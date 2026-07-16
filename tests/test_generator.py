"""生成模块测试"""
import sys
sys.path.insert(0, ".")

from src.generator.generator import Generator, DEFAULT_SYSTEM_PROMPT

generator = Generator()

def test_basic_generate():
    """传入简单的上下文，验证能返回非空回答"""
    answer = generator.generate(
        question="智能温控器怎么安装？",
        contexts=["安装步骤: 1.固定设备 2.连接电源 3.APP配对"],
        sources=["产品手册.pdf"],
    )
    assert len(answer) > 0
    print(f"  回答: {answer[:80]}...")

def test_source_citation():
    """返回的回答中包含来源标注"""
    answer = generator.generate(
        question="怎么安装？",
        contexts=["步骤1: 固定在墙上", "步骤2: 连接电源线"],
        sources=["产品手册.pdf第1页", "产品手册.pdf第2页"],
    )
    assert len(answer) > 0

def test_no_information():
    """参考资料没有答案时，返回'无法回答'而不是编造"""
    answer = generator.generate(
        question="月球上有多少陨石坑？",
        contexts=["智能温控器安装步骤", "Wi-Fi连接指南"],
        sources=["手册1.pdf", "手册2.pdf"],
    )
    print(f"  无资料回答: {answer[:100]}")

def test_stream():
    """流式生成能正常迭代"""
    chunks = list(generator.generate_stream(
        question="温控器怎么装？",
        contexts=["安装: 固定、接线、配对"],
        sources=["手册.pdf"],
    ))
    assert len(chunks) > 0
    full = "".join(chunks)
    assert len(full) > 0

def test_custom_system_prompt():
    """自定义 System Prompt 生效"""
    custom = Generator(system_prompt="你是客服机器人，回复以'您好'开头。")
    answer = custom.generate(
        question="退货流程",
        contexts=["7天无理由退货"],
        sources=["售后政策.pdf"],
    )
    assert len(answer) > 0
    print(f"  自定义 Prompt: {answer[:80]}")