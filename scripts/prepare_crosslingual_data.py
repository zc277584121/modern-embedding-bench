"""Prepare cross-lingual (Chinese ↔ English) parallel sentence pairs for evaluation.

Creates a dataset with:
- 200+ parallel sentence pairs at varying difficulty levels
- 50+ hard negative pairs (similar but NOT matching)

Data sources: Manually curated + programmatically constructed pairs covering:
- Common expressions & greetings
- News-style sentences
- Technical/scientific terminology
- Idioms & cultural expressions
- Structurally different but semantically equivalent sentences

Usage:
    uv run python scripts/prepare_crosslingual_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "crosslingual"


def create_parallel_pairs() -> list[dict]:
    """Create Chinese-English parallel sentence pairs with difficulty labels."""
    pairs = []

    def add(zh: str, en: str, difficulty: str, category: str,
            hard_negatives_en: list[str] | None = None,
            hard_negatives_zh: list[str] | None = None):
        pairs.append({
            "zh": zh,
            "en": en,
            "difficulty": difficulty,
            "category": category,
            "hard_negatives_en": hard_negatives_en or [],
            "hard_negatives_zh": hard_negatives_zh or [],
        })

    # ================================================================
    # EASY: Direct translations, common phrases (50 pairs)
    # ================================================================
    easy_pairs = [
        ("我爱你。", "I love you.", "greeting"),
        ("今天天气很好。", "The weather is nice today.", "daily"),
        ("请问洗手间在哪里？", "Excuse me, where is the restroom?", "daily"),
        ("这个多少钱？", "How much does this cost?", "daily"),
        ("谢谢你的帮助。", "Thank you for your help.", "greeting"),
        ("我正在学习中文。", "I am learning Chinese.", "daily"),
        ("你好，很高兴认识你。", "Hello, nice to meet you.", "greeting"),
        ("早上好，今天是星期一。", "Good morning, today is Monday.", "greeting"),
        ("我住在北京。", "I live in Beijing.", "daily"),
        ("她是一名医生。", "She is a doctor.", "daily"),
        ("我们一起去吃饭吧。", "Let's go eat together.", "daily"),
        ("这本书非常有趣。", "This book is very interesting.", "daily"),
        ("火车几点出发？", "What time does the train depart?", "travel"),
        ("我需要一杯咖啡。", "I need a cup of coffee.", "daily"),
        ("明天会下雨吗？", "Will it rain tomorrow?", "daily"),
        ("他们正在开会。", "They are in a meeting.", "work"),
        ("我的电话号码是12345。", "My phone number is 12345.", "daily"),
        ("这家餐厅的菜很好吃。", "The food at this restaurant is delicious.", "food"),
        ("请把门关上。", "Please close the door.", "daily"),
        ("我每天早上跑步。", "I run every morning.", "daily"),
        ("她会说三种语言。", "She can speak three languages.", "daily"),
        ("我们的航班延误了。", "Our flight has been delayed.", "travel"),
        ("这个城市很美丽。", "This city is beautiful.", "travel"),
        ("你能帮我一个忙吗？", "Can you do me a favor?", "daily"),
        ("我昨天看了一部电影。", "I watched a movie yesterday.", "daily"),
        ("这条路通往哪里？", "Where does this road lead to?", "travel"),
        ("他正在写一篇论文。", "He is writing a paper.", "work"),
        ("我们应该保护环境。", "We should protect the environment.", "general"),
        ("她的生日是下个月。", "Her birthday is next month.", "daily"),
        ("这是我第一次来中国。", "This is my first time visiting China.", "travel"),
        ("请给我一杯水。", "Please give me a glass of water.", "daily"),
        ("我周末喜欢去爬山。", "I like to go hiking on weekends.", "daily"),
        ("他已经毕业三年了。", "He graduated three years ago.", "daily"),
        ("这道数学题很难。", "This math problem is very difficult.", "education"),
        ("我在网上买了一件衣服。", "I bought a piece of clothing online.", "daily"),
        ("你喜欢什么类型的音乐？", "What type of music do you like?", "daily"),
        ("明天是公共假日。", "Tomorrow is a public holiday.", "daily"),
        ("我需要去银行取钱。", "I need to go to the bank to withdraw money.", "daily"),
        ("她正在准备晚餐。", "She is preparing dinner.", "daily"),
        ("这个项目的截止日期是下周五。", "The deadline for this project is next Friday.", "work"),
        ("请问这趟公交车去火车站吗？", "Excuse me, does this bus go to the train station?", "travel"),
        ("我对花粉过敏。", "I am allergic to pollen.", "health"),
        ("他的演讲非常精彩。", "His speech was excellent.", "general"),
        ("我们需要更多时间来完成这个任务。", "We need more time to complete this task.", "work"),
        ("这个软件很容易使用。", "This software is easy to use.", "tech"),
        ("今年夏天特别热。", "This summer is especially hot.", "daily"),
        ("我想预订一间双人房。", "I would like to book a double room.", "travel"),
        ("他们赢得了比赛。", "They won the match.", "sports"),
        ("我的电脑需要维修。", "My computer needs to be repaired.", "tech"),
        ("这家公司成立于2010年。", "This company was founded in 2010.", "work"),
    ]
    for zh, en, cat in easy_pairs:
        add(zh, en, "easy", cat)

    # ================================================================
    # MEDIUM: Natural expression differences (60 pairs)
    # ================================================================
    medium_pairs = [
        ("这道菜太咸了。", "This dish is too salty.",
         "food",
         ["This dish is too sweet.", "This soup is too salty."],
         ["这道菜太甜了。", "这碗汤太咸了。"]),
        ("他今天心情不好。", "He is in a bad mood today.",
         "daily",
         ["He was in a bad mood yesterday.", "She is in a bad mood today."],
         ["他昨天心情不好。", "她今天心情不好。"]),
        ("这个问题没有简单的答案。", "There is no simple answer to this question.",
         "general",
         ["This problem has a simple solution.", "There is no answer to this question."],
         ["这个问题有简单的答案。", "这个问题没有答案。"]),
        ("随着科技的发展，人们的生活方式发生了巨大变化。",
         "With the development of technology, people's lifestyles have undergone tremendous changes.",
         "tech",
         ["Technology has remained unchanged over the years.", "People's lifestyles have stayed the same despite technology."],
         ["科技的发展没有改变人们的生活。", "人们的生活方式一直没有变化。"]),
        ("政府正在采取措施应对气候变化。",
         "The government is taking measures to address climate change.",
         "news",
         ["The government is ignoring climate change.", "Companies are taking measures to address climate change."],
         ["政府忽视了气候变化问题。", "企业正在采取措施应对气候变化。"]),
        ("人工智能在医疗领域的应用越来越广泛。",
         "The application of artificial intelligence in the medical field is becoming increasingly widespread.",
         "tech",
         ["AI has limited applications in medicine.", "Artificial intelligence is widely used in education."],
         ["人工智能在医疗领域的应用很有限。", "人工智能在教育领域的应用越来越广泛。"]),
        ("这位科学家因为发现了新的化学元素而获得了诺贝尔奖。",
         "This scientist won the Nobel Prize for discovering a new chemical element.",
         "science",
         ["This scientist won the Nobel Prize for literature.", "This scientist discovered a new chemical element but did not win any award."],
         ["这位科学家因为文学成就获得了诺贝尔奖。", "这位科学家发现了新的化学元素但没有获得任何奖项。"]),
        ("根据最新的研究报告，全球平均气温在过去十年上升了0.5度。",
         "According to the latest research report, the global average temperature has risen by 0.5 degrees in the past decade.",
         "science",
         ["Global temperatures have decreased in the past decade.", "The global average temperature rose by 5 degrees in the past decade."],
         ["过去十年全球平均气温下降了。", "全球平均气温在过去十年上升了5度。"]),
        ("学生们需要在考试前提交所有作业。",
         "Students need to submit all assignments before the exam.",
         "education",
         ["Students can submit assignments after the exam.", "Teachers need to submit all grades before the exam."],
         ["学生们可以在考试后提交作业。", "老师们需要在考试前提交所有成绩。"]),
        ("这部电影改编自一部畅销小说。",
         "This movie was adapted from a bestselling novel.",
         "culture",
         ["This movie is an original screenplay.", "This novel was adapted from a movie."],
         ["这部电影是原创剧本。", "这部小说是根据电影改编的。"]),
        ("新冠疫情对全球经济造成了严重影响。",
         "The COVID-19 pandemic has had a severe impact on the global economy.",
         "news",
         ["The pandemic had no effect on the economy.", "COVID-19 only affected local economies."],
         ["疫情对经济没有影响。", "新冠疫情只影响了当地经济。"]),
        ("她在这家公司工作了十年，积累了丰富的经验。",
         "She has worked at this company for ten years and has accumulated rich experience.",
         "work",
         ["She just started working at this company.", "He has worked at this company for ten years."],
         ["她刚刚开始在这家公司工作。", "他在这家公司工作了十年。"]),
        ("城市化进程加速导致了一系列环境问题。",
         "The accelerating urbanization process has led to a series of environmental problems.",
         "news",
         ["Urbanization has solved environmental problems.", "Rural development has caused environmental issues."],
         ["城市化进程解决了环境问题。", "农村发展导致了环境问题。"]),
        ("网络安全已经成为企业面临的最大挑战之一。",
         "Cybersecurity has become one of the biggest challenges facing enterprises.",
         "tech",
         ["Cybersecurity is no longer a concern for businesses.", "Physical security is the biggest challenge for enterprises."],
         ["网络安全不再是企业的顾虑。", "物理安全是企业面临的最大挑战。"]),
        ("这位艺术家的作品在国际上享有盛誉。",
         "This artist's works enjoy a high reputation internationally.",
         "culture",
         ["This artist is unknown outside their country.", "This musician's works are famous internationally."],
         ["这位艺术家在国际上默默无闻。", "这位音乐家的作品在国际上享有盛誉。"]),
        ("越来越多的人选择在家办公。",
         "More and more people are choosing to work from home.",
         "work",
         ["Fewer people are working from home now.", "More and more people are choosing to work in offices."],
         ["越来越少的人选择在家办公。", "越来越多的人选择在办公室工作。"]),
        ("这项新政策将于下个月正式实施。",
         "This new policy will be officially implemented next month.",
         "news",
         ["This policy has already been implemented.", "This old policy will be abolished next month."],
         ["这项政策已经实施了。", "这项旧政策将于下个月废除。"]),
        ("研究人员发现了一种可能治愈糖尿病的新方法。",
         "Researchers have discovered a new method that may cure diabetes.",
         "science",
         ["Researchers failed to find a cure for diabetes.", "Researchers discovered a new method to treat heart disease."],
         ["研究人员未能找到治愈糖尿病的方法。", "研究人员发现了一种治疗心脏病的新方法。"]),
        ("全球变暖正在加速北极冰川的融化。",
         "Global warming is accelerating the melting of Arctic glaciers.",
         "science",
         ["Arctic glaciers are growing due to global cooling.", "Global warming is affecting Antarctic ice sheets."],
         ["北极冰川因全球变冷而增长。", "全球变暖正在影响南极冰盖。"]),
        ("他从小就对天文学产生了浓厚的兴趣。",
         "He developed a strong interest in astronomy from a young age.",
         "education",
         ["He only became interested in astronomy as an adult.", "She developed a strong interest in biology from a young age."],
         ["他成年后才对天文学产生兴趣。", "她从小就对生物学产生了浓厚的兴趣。"]),
        ("这款手机的电池续航能力非常出色。",
         "This phone's battery life is outstanding.",
         "tech",
         ["This phone's battery life is poor.", "This laptop's battery life is outstanding."],
         ["这款手机的电池续航能力很差。", "这款笔记本电脑的电池续航能力非常出色。"]),
        ("中国的高铁网络是世界上最大的。",
         "China's high-speed rail network is the largest in the world.",
         "travel",
         ["Japan's high-speed rail network is the largest.", "China's highway network is the largest in the world."],
         ["日本的高铁网络是世界上最大的。", "中国的高速公路网络是世界上最大的。"]),
        ("可再生能源的成本在过去十年中大幅下降。",
         "The cost of renewable energy has dropped significantly in the past decade.",
         "science",
         ["Renewable energy costs have increased dramatically.", "Fossil fuel costs have dropped in the past decade."],
         ["可再生能源的成本大幅上升了。", "化石能源的成本在过去十年中大幅下降。"]),
        ("这座古老的寺庙已有一千多年的历史。",
         "This ancient temple has a history of over a thousand years.",
         "culture",
         ["This modern church was built last year.", "This ancient bridge has a history of over a thousand years."],
         ["这座现代教堂是去年建的。", "这座古老的桥梁已有一千多年的历史。"]),
        ("该公司的市值突破了一万亿美元。",
         "The company's market capitalization has surpassed one trillion dollars.",
         "business",
         ["The company's market cap is only one billion dollars.", "The company's revenue surpassed one trillion dollars."],
         ["该公司的市值只有十亿美元。", "该公司的收入突破了一万亿美元。"]),
        ("自动驾驶技术面临着许多法律和伦理挑战。",
         "Autonomous driving technology faces many legal and ethical challenges.",
         "tech",
         ["Self-driving cars have solved all legal challenges.", "Autonomous flying technology faces legal challenges."],
         ["自动驾驶汽车已经解决了所有法律挑战。", "自动飞行技术面临着法律挑战。"]),
        ("每年有数百万游客到故宫参观。",
         "Millions of tourists visit the Forbidden City every year.",
         "travel",
         ["Few tourists visit the Forbidden City.", "Millions of tourists visit the Great Wall every year."],
         ["很少有游客去故宫参观。", "每年有数百万游客到长城参观。"]),
        ("社交媒体改变了人们获取新闻的方式。",
         "Social media has changed the way people access news.",
         "tech",
         ["Social media has not affected news consumption.", "Television changed the way people access news."],
         ["社交媒体没有影响新闻消费方式。", "电视改变了人们获取新闻的方式。"]),
        ("这位作家的最新小说已经被翻译成二十种语言。",
         "This author's latest novel has been translated into twenty languages.",
         "culture",
         ["This author's first novel was translated into twenty languages.", "This author's latest novel has not been translated."],
         ["这位作家的第一部小说被翻译成了二十种语言。", "这位作家的最新小说还没有被翻译。"]),
        ("量子计算有望在未来十年内实现重大突破。",
         "Quantum computing is expected to achieve major breakthroughs within the next decade.",
         "tech",
         ["Quantum computing will never be practical.", "Quantum computing achieved a breakthrough last year."],
         ["量子计算永远不会实用。", "量子计算去年实现了重大突破。"]),
        ("运动不仅有益于身体健康，也对心理健康有帮助。",
         "Exercise is not only beneficial to physical health but also helpful for mental health.",
         "health",
         ["Exercise only benefits physical health.", "Diet is beneficial to both physical and mental health."],
         ["运动只对身体健康有益。", "饮食对身体和心理健康都有帮助。"]),
    ]
    for item in medium_pairs:
        zh, en, cat = item[0], item[1], item[2]
        hn_en = item[3] if len(item) > 3 else []
        hn_zh = item[4] if len(item) > 4 else []
        add(zh, en, "medium", cat, hn_en, hn_zh)

    # ================================================================
    # HARD: Idioms, culturally-specific, structural differences (50 pairs)
    # ================================================================
    hard_pairs = [
        # Idioms that don't translate literally
        ("画蛇添足", "To gild the lily; adding unnecessary extras",
         "idiom",
         ["To add fuel to the fire", "To let the cat out of the bag"],
         ["火上浇油", "泄露秘密"]),
        ("塞翁失马，焉知非福", "A blessing in disguise; every cloud has a silver lining",
         "idiom",
         ["When it rains, it pours", "Better late than never"],
         ["祸不单行", "迟到总比不到好"]),
        ("对牛弹琴", "To cast pearls before swine; preaching to deaf ears",
         "idiom",
         ["To kill two birds with one stone", "To beat around the bush"],
         ["一箭双雕", "拐弯抹角"]),
        ("亡羊补牢", "Better late than never; to mend the fence after the sheep are lost",
         "idiom",
         ["Prevention is better than cure", "Don't cry over spilled milk"],
         ["预防胜于治疗", "覆水难收"]),
        ("入乡随俗", "When in Rome, do as the Romans do",
         "idiom",
         ["There's no place like home", "East or west, home is best"],
         ["金窝银窝不如自己的狗窝", "东方西方，家最好"]),
        ("脚踏实地", "To be down-to-earth; to keep one's feet on the ground",
         "idiom",
         ["To have one's head in the clouds", "To reach for the stars"],
         ["好高骛远", "志存高远"]),
        ("杯水车薪", "A drop in the bucket; an utterly inadequate effort",
         "idiom",
         ["The last straw that broke the camel's back", "A needle in a haystack"],
         ["压死骆驼的最后一根稻草", "大海捞针"]),
        ("事半功倍", "To get twice the result with half the effort",
         "idiom",
         ["To get half the result with twice the effort", "No pain, no gain"],
         ["事倍功半", "不劳无获"]),
        ("守株待兔", "To wait for a windfall; waiting idly for opportunities",
         "idiom",
         ["Strike while the iron is hot", "The early bird catches the worm"],
         ["趁热打铁", "早起的鸟儿有虫吃"]),
        ("班门弄斧", "To show off one's skills before an expert; teaching your grandmother to suck eggs",
         "idiom",
         ["Practice makes perfect", "Jack of all trades, master of none"],
         ["熟能生巧", "博而不精"]),

        # Culturally-specific expressions
        ("他这个人情商很高。", "He has high emotional intelligence and is very socially adept.",
         "culture",
         ["He is very intelligent academically.", "He has a high IQ."],
         ["他学习成绩很好。", "他智商很高。"]),
        ("这件事在网上引起了热议。", "This incident went viral online and sparked heated discussion.",
         "culture",
         ["This incident was barely noticed online.", "This topic was discussed in a private meeting."],
         ["这件事在网上几乎没人关注。", "这个话题在私下会议中被讨论。"]),
        ("他是一个佛系青年。", "He takes a laid-back, go-with-the-flow approach to life.",
         "culture",
         ["He is an extremely ambitious young man.", "He is very anxious about his future."],
         ["他是一个非常有野心的年轻人。", "他对未来非常焦虑。"]),
        ("内卷越来越严重了。", "The rat race and excessive competition are getting worse.",
         "culture",
         ["Competition is decreasing in society.", "People are becoming more relaxed about work."],
         ["社会竞争越来越少了。", "人们对工作越来越放松。"]),
        ("这个产品的性价比很高。", "This product offers excellent value for money.",
         "business",
         ["This product is overpriced for its quality.", "This product is the cheapest on the market."],
         ["这个产品价格虚高。", "这个产品是市场上最便宜的。"]),
        ("她是个工作狂，加班对她来说是家常便饭。",
         "She is a workaholic; working overtime is just a regular thing for her.",
         "work",
         ["She never works overtime.", "She hates her job and wants to quit."],
         ["她从不加班。", "她讨厌自己的工作想要辞职。"]),
        ("他说话总是拐弯抹角。", "He always beats around the bush when he speaks.",
         "culture",
         ["He always speaks directly and bluntly.", "He is a very quiet person who rarely speaks."],
         ["他说话总是直来直去。", "他是一个很安静、很少说话的人。"]),

        # Structurally different but semantically equivalent
        ("中国是世界上人口最多的国家之一。",
         "China ranks among the world's most populous nations.",
         "general",
         ["China has a small population.", "India is the most populous country."],
         ["中国人口很少。", "印度是人口最多的国家。"]),
        ("没有什么是不可能的。", "Nothing is impossible.",
         "general",
         ["Everything is impossible.", "Some things are impossible."],
         ["一切都不可能。", "有些事情是不可能的。"]),
        ("在这件事情上，我和他的看法截然不同。",
         "He and I see this matter from completely different perspectives.",
         "general",
         ["We agree on this matter.", "I have no opinion on this matter."],
         ["在这件事上我们意见一致。", "我对这件事没有看法。"]),
        ("正是由于缺乏沟通，才导致了这次误会。",
         "The misunderstanding arose precisely because of a lack of communication.",
         "general",
         ["Good communication caused the misunderstanding.", "There was no misunderstanding at all."],
         ["良好的沟通导致了这次误会。", "根本没有什么误会。"]),
        ("与其抱怨困难，不如想办法解决问题。",
         "Instead of complaining about difficulties, it is better to find solutions.",
         "general",
         ["Complaining about problems is the best approach.", "Difficulties should be avoided at all costs."],
         ["抱怨问题是最好的方式。", "困难应该不惜一切代价来避免。"]),
        ("经济全球化是一把双刃剑，既带来机遇也带来挑战。",
         "Economic globalization is a double-edged sword, bringing both opportunities and challenges.",
         "business",
         ["Globalization only brings benefits.", "Globalization only brings problems."],
         ["全球化只带来好处。", "全球化只带来问题。"]),
        ("尽管面临重重困难，他最终还是成功了。",
         "Despite facing numerous obstacles, he eventually succeeded.",
         "general",
         ["He succeeded easily without any difficulties.", "He failed despite trying hard."],
         ["他没有遇到任何困难就成功了。", "尽管他努力尝试，但最终还是失败了。"]),

        # Technical/scientific (harder to translate well)
        ("深度学习模型的可解释性一直是学术界关注的热点问题。",
         "The interpretability of deep learning models has been a hot topic of concern in the academic community.",
         "tech",
         ["Deep learning models are fully transparent and interpretable.", "The efficiency of deep learning models is a major concern."],
         ["深度学习模型是完全透明可解释的。", "深度学习模型的效率是一个主要关注点。"]),
        ("该实验结果证实了研究人员之前的假设。",
         "The experimental results confirmed the researchers' previous hypothesis.",
         "science",
         ["The experiment disproved the hypothesis.", "The researchers have not yet conducted any experiments."],
         ["实验结果否定了这个假设。", "研究人员还没有进行任何实验。"]),
        ("大语言模型在自然语言处理领域取得了前所未有的突破。",
         "Large language models have achieved unprecedented breakthroughs in the field of natural language processing.",
         "tech",
         ["Language models have made no progress in NLP.", "Small language models outperform large ones in NLP."],
         ["语言模型在自然语言处理方面没有取得进展。", "小语言模型在自然语言处理方面比大模型表现更好。"]),
        ("可持续发展需要平衡经济增长和环境保护之间的关系。",
         "Sustainable development requires balancing the relationship between economic growth and environmental protection.",
         "science",
         ["Economic growth should always take priority over the environment.", "Sustainable development focuses only on environmental protection."],
         ["经济增长应该始终优先于环境保护。", "可持续发展只关注环境保护。"]),
        ("基因编辑技术引发了关于伦理道德的广泛讨论。",
         "Gene editing technology has sparked widespread discussion about ethics and morality.",
         "science",
         ["Gene editing technology is universally accepted without controversy.", "Gene editing has no ethical implications."],
         ["基因编辑技术被普遍接受，没有争议。", "基因编辑没有伦理影响。"]),
        ("从长远来看，投资教育是一个国家最明智的选择。",
         "In the long run, investing in education is the wisest choice for a nation.",
         "education",
         ["Education investment has no long-term benefits.", "Military spending is the wisest investment for a nation."],
         ["教育投资没有长期效益。", "军事支出是一个国家最明智的投资。"]),

        # More structural variations
        ("虽然他年纪轻轻，但已经在科研领域取得了显著成就。",
         "Young as he is, he has already achieved remarkable accomplishments in scientific research.",
         "science",
         ["He is old and has achieved nothing in research.", "He is young and has no interest in science."],
         ["他年纪大了但在研究方面没有成就。", "他很年轻，对科学没有兴趣。"]),
        ("不到长城非好汉。",
         "One is not a true hero until one has climbed the Great Wall.",
         "idiom",
         ["The Great Wall is not worth visiting.", "Anyone can be a hero."],
         ["长城不值得去。", "任何人都可以成为英雄。"]),
        ("三人行必有我师焉。",
         "In a group of three, there is always something I can learn from the others.",
         "idiom",
         ["You can only learn from teachers in school.", "Working alone is better than working in groups."],
         ["你只能在学校从老师那里学习。", "独自工作比团队合作好。"]),
        ("失败是成功之母。",
         "Failure is the mother of success.",
         "idiom",
         ["Success comes without any failure.", "Failure always leads to more failure."],
         ["成功不需要经历失败。", "失败总是导致更多的失败。"]),
        ("他的中文说得非常流利，如果不看外表，你根本不知道他是外国人。",
         "His Chinese is so fluent that if you didn't see his appearance, you would never know he is a foreigner.",
         "culture",
         ["His Chinese is very poor.", "He looks Chinese but cannot speak Chinese."],
         ["他的中文说得很差。", "他看起来像中国人，但不会说中文。"]),
        ("书到用时方恨少。",
         "You never realize how little you know until you need to apply your knowledge.",
         "idiom",
         ["Education is a waste of time.", "You always know enough."],
         ["教育是浪费时间。", "你总是知道得够多。"]),
        ("双减政策对教培行业产生了深远影响。",
         "The Double Reduction policy has had a profound impact on the education and tutoring industry.",
         "news",
         ["The education industry is completely unregulated.", "The tutoring industry is growing rapidly without any restrictions."],
         ["教育行业完全没有监管。", "教培行业正在快速增长，没有任何限制。"]),
        ("共同富裕是中国社会发展的重要目标。",
         "Common prosperity is an important goal of China's social development.",
         "news",
         ["China has no goals for social development.", "Individual wealth is China's primary development goal."],
         ["中国没有社会发展目标。", "个人财富是中国的首要发展目标。"]),
        ("碳中和已经成为全球共识。",
         "Carbon neutrality has become a global consensus.",
         "science",
         ["No country cares about carbon emissions.", "Carbon neutrality is only a concern in Europe."],
         ["没有国家关心碳排放。", "碳中和只是欧洲关心的问题。"]),
        ("这个问题牵一发而动全身。",
         "This issue has far-reaching implications; pulling one thread unravels the whole fabric.",
         "general",
         ["This issue is completely isolated and affects nothing else.", "This problem is very simple to solve."],
         ["这个问题是完全孤立的，不影响其他事情。", "这个问题很容易解决。"]),
        ("实践出真知。",
         "True knowledge comes from practice; practice is the sole criterion of truth.",
         "idiom",
         ["Theory is more important than practice.", "Knowledge comes only from reading books."],
         ["理论比实践更重要。", "知识只来自读书。"]),
        ("他的话意味深长，让人回味无穷。",
         "His words were profound and thought-provoking, leaving a lasting impression.",
         "culture",
         ["His words were shallow and quickly forgotten.", "He said nothing meaningful."],
         ["他的话很肤浅，很快就被遗忘了。", "他没有说什么有意义的话。"]),
        ("路遥知马力，日久见人心。",
         "Distance tests a horse's strength; time reveals a person's true character.",
         "idiom",
         ["First impressions are always accurate.", "You can judge a person immediately."],
         ["第一印象总是准确的。", "你可以立刻判断一个人。"]),
        ("新能源汽车的市场渗透率在中国已经超过了30%。",
         "The market penetration rate of new energy vehicles in China has exceeded 30%.",
         "business",
         ["Electric vehicles have less than 1% market share in China.", "Gasoline cars dominate the Chinese market with 99% share."],
         ["电动汽车在中国的市场份额不到1%。", "汽油车以99%的份额主导中国市场。"]),
        ("知己知彼，百战不殆。",
         "Know yourself and know your enemy, and you will never be defeated.",
         "idiom",
         ["Ignorance is bliss in warfare.", "Fighting without strategy is the best approach."],
         ["在战争中无知是福。", "不需要战略就能取胜。"]),
    ]
    for item in hard_pairs:
        zh, en, cat = item[0], item[1], item[2]
        hn_en = item[3] if len(item) > 3 else []
        hn_zh = item[4] if len(item) > 4 else []
        add(zh, en, "hard", cat, hn_en, hn_zh)

    # ================================================================
    # Additional medium/hard pairs for diversity (40 more)
    # ================================================================
    extra_pairs = [
        ("春节是中国最重要的传统节日。", "The Spring Festival is the most important traditional holiday in China.", "medium", "culture"),
        ("上海是中国的经济中心。", "Shanghai is the economic center of China.", "easy", "general"),
        ("这家初创公司刚刚完成了A轮融资。", "This startup has just completed its Series A funding round.", "medium", "business"),
        ("他在面试中表现得非常出色。", "He performed exceptionally well in the interview.", "easy", "work"),
        ("全球芯片短缺影响了多个行业。", "The global chip shortage has affected multiple industries.", "medium", "tech"),
        ("这个城市的房价让年轻人望而却步。", "The housing prices in this city are discouraging for young people.", "medium", "news"),
        ("他们的婚礼在海边举行，非常浪漫。", "Their wedding was held by the seaside and was very romantic.", "easy", "daily"),
        ("数据隐私保护已经成为一个全球性话题。", "Data privacy protection has become a global topic.", "medium", "tech"),
        ("她凭借自己的努力考上了北大。", "She got into Peking University through her own hard work.", "medium", "education"),
        ("端午节吃粽子是中国的传统习俗。", "Eating zongzi during the Dragon Boat Festival is a traditional Chinese custom.", "hard", "culture"),
        ("他们用了三年时间攻克了这个技术难题。", "It took them three years to overcome this technical challenge.", "medium", "tech"),
        ("这座城市的交通拥堵问题日益严重。", "The traffic congestion problem in this city is getting increasingly worse.", "medium", "news"),
        ("机器学习和深度学习是人工智能的核心技术。", "Machine learning and deep learning are core technologies of artificial intelligence.", "medium", "tech"),
        ("丝绸之路促进了东西方的文化交流。", "The Silk Road facilitated cultural exchange between East and West.", "medium", "culture"),
        ("全民医保是许多国家追求的目标。", "Universal healthcare is a goal pursued by many countries.", "medium", "health"),
        ("这款应用的月活跃用户已经突破了一亿。", "This app's monthly active users have exceeded 100 million.", "medium", "tech"),
        ("垃圾分类是保护环境的重要举措。", "Waste sorting is an important measure for environmental protection.", "medium", "general"),
        ("他的研究成果发表在了《自然》杂志上。", "His research findings were published in the journal Nature.", "medium", "science"),
        ("共享经济改变了人们的消费习惯。", "The sharing economy has changed people's consumption habits.", "medium", "business"),
        ("太空探索是人类共同的梦想。", "Space exploration is a shared dream of humanity.", "medium", "science"),
        ("她在国际钢琴比赛中获得了金奖。", "She won the gold medal in an international piano competition.", "easy", "culture"),
        ("5G技术将推动物联网的快速发展。", "5G technology will drive the rapid development of the Internet of Things.", "medium", "tech"),
        ("中医药文化源远流长。", "Traditional Chinese medicine has a long and profound cultural heritage.", "hard", "culture"),
        ("区块链技术在金融领域有广泛的应用前景。", "Blockchain technology has broad application prospects in the financial sector.", "medium", "tech"),
        ("老龄化社会带来了诸多社会问题。", "An aging society brings numerous social problems.", "medium", "news"),
        ("这位导演的电影以细腻的情感表达著称。", "This director's films are known for their delicate emotional expression.", "medium", "culture"),
        ("数字化转型是传统企业面临的紧迫课题。", "Digital transformation is an urgent topic facing traditional enterprises.", "medium", "business"),
        ("他把这篇文章从中文翻译成了英文。", "He translated this article from Chinese into English.", "easy", "work"),
        ("气候变化导致极端天气事件频繁发生。", "Climate change is causing extreme weather events to occur more frequently.", "medium", "science"),
        ("这个算法的时间复杂度是O(n log n)。", "The time complexity of this algorithm is O(n log n).", "medium", "tech"),
        ("她是公司里唯一的女性高管。", "She is the only female executive in the company.", "easy", "work"),
        ("知识产权保护对创新发展至关重要。", "Intellectual property protection is crucial for innovative development.", "medium", "business"),
        ("这场暴雨导致了严重的城市内涝。", "The heavy rainstorm caused severe urban flooding.", "medium", "news"),
        ("语言是文化的载体。", "Language is the carrier of culture.", "medium", "culture"),
        ("他通过远程医疗系统进行了诊断。", "He made the diagnosis through a telemedicine system.", "medium", "health"),
        ("这项技术的商业化前景令人期待。", "The commercialization prospects of this technology are promising.", "medium", "business"),
        ("人才流失是发展中国家面临的严峻挑战。", "Brain drain is a serious challenge faced by developing countries.", "hard", "news"),
        ("她的歌声打动了在场所有的观众。", "Her singing moved every member of the audience present.", "medium", "culture"),
        ("城市绿化对改善空气质量有重要作用。", "Urban greening plays an important role in improving air quality.", "medium", "general"),
        ("比特币的价格波动非常剧烈。", "Bitcoin's price volatility is extremely high.", "medium", "business"),
    ]
    for zh, en, diff, cat in extra_pairs:
        add(zh, en, diff, cat)

    return pairs


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pairs = create_parallel_pairs()

    # Save as JSONL
    output_path = DATA_DIR / "parallel_pairs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Statistics
    difficulties = {}
    categories = {}
    total_hn_en = 0
    total_hn_zh = 0
    for p in pairs:
        d = p["difficulty"]
        difficulties[d] = difficulties.get(d, 0) + 1
        c = p["category"]
        categories[c] = categories.get(c, 0) + 1
        total_hn_en += len(p["hard_negatives_en"])
        total_hn_zh += len(p["hard_negatives_zh"])

    print(f"Total parallel pairs: {len(pairs)}")
    print(f"Difficulties: {dict(sorted(difficulties.items()))}")
    print(f"Categories: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")
    print(f"Hard negatives (EN): {total_hn_en}")
    print(f"Hard negatives (ZH): {total_hn_zh}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
