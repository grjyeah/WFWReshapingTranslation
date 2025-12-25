#!/usr/bin/env python3
"""
测试带时间戳的文件保存功能
"""
import sys
sys.path.insert(0, '/home/aipcuser/pyworkspaces/WFWReshapingTranslation')
from meeting_processor import MeetingTranscriptProcessor

def test_timestamp_save():
    """测试时间戳文件名生成"""

    processor = MeetingTranscriptProcessor()

    # 测试时间戳生成
    test_names = [
        "processed_chinese.txt",
        "english_translation.txt",
        "test.txt"
    ]

    print("=" * 80)
    print("🧪 测试时间戳文件名生成")
    print("=" * 80)

    for name in test_names:
        timestamped = processor._generate_timestamped_filename(name)
        print(f"\n{name} → {timestamped}")

    print("\n" + "=" * 80)

    # 测试文件保存
    test_text = """[主持人]：大家好，欢迎参加今天的会议。
[张总]：我们需要建立一个完善的数据治理体系。"""

    print("\n🔄 测试文件保存...")
    print("-" * 80)

    result = processor.process_and_translate(test_text, save_intermediate=True)

    print("\n" + "=" * 80)
    print("✓ 测试完成！请检查 processed/ 文件夹")
    print("=" * 80)

    # 列出processed文件夹中的文件
    import os
    if os.path.exists("processed"):
        files = os.listdir("processed")
        print(f"\n📁 processed/ 文件夹内容：")
        for f in sorted(files):
            filepath = f"processed/{f}"
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  - {f} ({size} 字节)")

if __name__ == "__main__":
    test_timestamp_save()
