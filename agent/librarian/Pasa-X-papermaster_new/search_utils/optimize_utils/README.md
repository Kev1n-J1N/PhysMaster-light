1. call_ar5iv   调用ar5iv的接口     根据id搜索sections
2. call_db      调用数据库的接口    根据title搜索arxiv_id（db1/2还可以获取sections）
3. call_llm     调用llm的接口       输入sys/usr prompt，返回模型回答
4. parse_utils  解析tex文件的工具函数
5. paper_cache/ 解析时临时存储tex文件
6. sim_title    简化标题，便于比对
7. fetch_xxx    用于并行测试单个工具（可能不适配）
8. test_xxx     用于并行测试完整工具链（可能不适配）
9. gen_xxx      生成测试/数据库数据
10. paraline     并行类，提供优雅退出逻辑
11. wechat_utils.py   微信通知工具（缺少psw文件）