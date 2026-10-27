"""Requirement registry: clause (regulation_requirements.csv) -> requirement (需求规格.md) -> tests.

clauses   keys "source|clause" that must exist in regulation_requirements.csv
da        design-assumption tags (DA-xx) that must exist in dms/config.py PROVENANCE
tests     selectors over result IDs; a trailing "*" is a prefix match.  Result IDs come from
          results/regulation_tests.csv (PROPOSED rows), results/boundary_tests.csv, FA-<config>
          (false_alarm_summary.csv), NH-<variant> (nhtsa_summary.csv) and DD-SYN.  Unit tests link
          themselves to requirements with a "REQ: ID, ID" line in their docstring.
na        if a requirement is not verified in this project: why, and where it would be verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GSR, ADDW, DDAW = "EU 2019/2144 GSR", "EU 2023/2590 ADDW", "EU 2021/1341 DDAW"
DE, SD = "Euro NCAP DE v1.2", "Euro NCAP SD-202 v1.2"
AOV, ASA, IVA = "ASEAN NCAP Overall v3.1", "ASEAN NCAP SA v3.2", "ASEAN NCAP IVASA ICI-TP"
NH, GB = "NHTSA VM-DDG 2013", "GB/T 41797-2022"
P1, P2, P3 = "Annex I Part 1 ", "Annex I Part 2 ", "Annex I Part 3 "


def k(src: str, *clauses: str) -> list[str]:
    return [f"{src}|{c}" for c in clauses]


@dataclass(frozen=True)
class Req:
    rid: str
    title: str
    text: str
    kind: str
    clauses: list[str]
    da: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    na: str = ""


REQS: list[Req] = [
    # ------------------------------------------------------------------ applicability (document review)
    Req("APP-01", "欧盟强制装备与时间表", "M/N 类车辆须装 ADDW 与 DDAW：ADDW 新车型 2024-07-07、所有新车 2026-07-07；DDAW 法规 2022-07-06 起适用", "适用性",
        k(GSR, "Art. 3(5)", "Art. 3(6)", "Art. 6(1)(c)(d)", "Annex II E2/E3", "Annex II 表注 B/C")
        + k(ADDW, "Art. 1", "Art. 5", "解释备忘录 §1") + k(DDAW, "Art. 1", "Art. 6"),
        na="适用性条款，用文件审查确认（report.md 第 2 节），不做软件测试"),
    Req("APP-02", "ADDW 只管长时视觉分心", "ADDW 现阶段只要求长时单次注视；累计扫视（VATS）按 Euro NCAP 实现，作为高于法规的设计选择", "适用性",
        k(ADDW, "序言 (2)", "序言 (3)"), na="范围界定，文件审查；VATS 的验证见 DIS-05"),
    Req("APP-03", "Euro NCAP 2026 评分结构", "驾驶员参与度 30 分（监测 25 + 通用控制 5），作为功能取舍的依据", "适用性",
        k(DE, "评分表"), na="评分口径，文件审查；本项目不计算 Euro NCAP 得分"),
    Req("APP-04", "ASEAN NCAP 2026–2030 的实际要求", "DSM 只是 Advanced SAT 的一项功能定义，按装配给分（每项 1 分，合计上限 2 分），无性能试验；IVASA 为评级外项目", "适用性",
        k(AOV, "§1") + k(ASA, "§1/§2", "§7.2.4", "§7.3.1", "§7.4/§7.5") + k(IVA, "IVASA 页面"),
        na="评级规则，文件审查；IVASA 座舱试验另见 TST-04"),
    Req("APP-05", "中国 GB/T 41797-2022", "推荐性国家标准，现行；技术条款未核实，本项目不采用其数值", "适用性",
        k(GB, "标准信息", "技术条款"), na="全文未访问（在线阅读有验证码）；需取得正式文本后补做对标"),
    Req("APP-06", "NHTSA 指南的性质与范围", "非强制；第一阶段只覆盖视觉-手动交互，语音方案只能评估其残留的视觉-手动部分", "适用性",
        k(NH, "摘要/I"), na="范围界定，文件审查；评估本身见 TST-05 / HMT-01"),
    # ------------------------------------------------------------------ system
    Req("SYS-01", "激活车速", "车速 ≥20 km/h 时发出分心报警；低于 20 km/h 不报警但计时继续", "功能",
        k(ADDW, P1 + "§3.1.1", P1 + "§3.3.2.5–3.3.2.6") + k(DE, "§1.1 一般要求（车速）"),
        tests=["BC-SPD-19.9", "BC-SPD-20", "BC-SPD-20.1"]),
    Req("SYS-02", "默认开启与行程复位", "每次上电默认开启；驾驶员可关闭 ADDW 报警（不提供单次短按关闭），下一行程自动恢复", "功能",
        k(ADDW, P1 + "§3.1.2", P1 + "§3.1.6") + k(DDAW, P1 + "§3.1.1–3.1.3") + k(DE, "§1.1 一般要求（默认开启）")
        + k(SD, "§3.2.1"), tests=["SYS-JOURNEY-RESET"]),
    Req("SYS-03", "标定期", "上电后累计 20 s（≥10 km/h）完成标定，期间不发分心报警", "功能",
        k(ADDW, P1 + "§3.1.1") + k(DE, "§1.1 一般要求（开始测量）"), da=["DA-22"], tests=["SYS-CALIB"]),
    Req("SYS-04", "灵敏度不可调", "检测阈值为只读配置，运行时不可修改", "功能", k(DE, "§1.1 一般要求（默认开启）")),
    Req("SYS-05", "隐私与数据最小化", "不做身份识别、不用生物特征；只保留固定长度滚动缓冲（VATS 30 s、PERCLOS 60 s、微睡眠记录 5 min），不对外输出原始数据", "隐私",
        k(GSR, "Art. 6(3)") + k(ADDW, P1 + "§2.3.1–2.3.3") + k(DDAW, P1 + "§2.1/§2.3.1")),
    Req("SYS-06", "昼夜与环境条件", "白天、夜间及天气不限制时有效", "性能",
        k(ADDW, P1 + "§3.2.1", P2 + "§1.6") + k(DDAW, P1 + "§3.2.1–3.2.3"),
        na="取决于摄像头、红外补光和感知算法，逻辑层无法验证；需台架（照度 1–100000 lux）与昼夜实车试验"),
    Req("SYS-07", "误报最小化", "在自然驾驶中尽量不发不相关报警；用合成自然驾驶评估误报率与检测时延", "性能",
        k(ADDW, P1 + "§2.2") + k(SD, "§4.4"), da=["DA-19", "DA-23"], tests=["FA-*"]),
    Req("SYS-08", "其他系统接管时停用", "L3 等接管全部驾驶任务且有 DMS 时可停用 ADDW，条件消失后恢复", "功能",
        k(ADDW, P1 + "§3.1.3"), na="本项目没有自动驾驶/辅助驾驶接口；需在整车功能集成时验证"),
    Req("SYS-09", "故障时系统不停用", "故障期间系统保持运行（只停发分心报警），故障消失后自动恢复检测", "功能",
        k(ADDW, P1 + "§3.1.4"), tests=["SYS-FAILURE-NOLIGHT"]),
    # ------------------------------------------------------------------ distraction
    Req("DIS-01", "注视区域模型", "按 ADDW 几何定义（眼点、±55°、风挡+10°、下 30°）判定 Area 1/2/3，并规定区域到注视标签的映射", "功能",
        k(ADDW, P1 + "§3.3.1", P1 + "§3.3.1.1", P1 + "§3.3.1.2", P1 + "§3.3.1.3–3.3.1.4", P2 + "§1.4.2"), da=["DA-18"],
        tests=["ADDW-P2-*"]),
    Req("DIS-02", "长时单次注视", "非驾驶区域单次注视 ≥3.3 s（车速 ≥20 km/h，不分速度档、不使用非标称 +1.5 s 放宽）发出报警；申报 ΔT_LGA = 3.4 s", "功能",
        k(ADDW, P1 + "§2.1", P1 + "§3.3.2.1", P1 + "§3.3.2.2") + k(DE, "§1.2.1") + k(SD, "§1/§1.1"),
        da=["DA-04", "DA-02", "DA-01"],
        tests=["ADDW-P2-*", "ENCAP-LD-nondriving-*", "ENCAP-OCC-sunglasses-LD-owl", "IVASA-DIS-*", "BC-LD-*",
               "BC-ADDW-3.5", "BC-ADDWMIN-SPD-*", "BC-LAT-*", "FA-T_long=*"]),
    Req("DIS-03", "计时与车速无关", "激活期间注视计时持续进行，车速只门控报警", "功能", k(ADDW, P1 + "§3.3.2.3"), tests=["BC-SPD-CROSS"]),
    Req("DIS-04", "中断容差", "≤0.3 s 的中断（眨眼、伪影、进-出-进）不复位并计入；超过 0.3 s 的 unknown 冻结计时；其他标签超过 0.3 s 结束本次注视", "功能",
        k(ADDW, P1 + "§3.3.2.4"), da=["DA-03"], tests=["BC-GAP-*", "BC-BLINK", "BC-UNKNOWN-LONG", "FA-gap=*"]),
    Req("DIS-05", "累计扫视（VATS）", "30 s 窗内非驾驶注视累计 ≥10 s 报警；连续注视道路 ≥2.0 s 结束本轮累计", "功能",
        k(DE, "§1.2.2") + k(SD, "§4.1.1.2"), da=["DA-06"],
        tests=["ENCAP-VATS-nondriving-*", "ENCAP-VATS-multi-lizard", "BC-VATS-*", "BC-VATSRESET-*", "BC-VATSWIN-*",
               "FA-VATS=*", "FA-reset=*"]),
    Req("DIS-06", "手机使用", "phone 标签按 VATS 计入（位置不限，含举在风挡/仪表视野内）", "功能",
        k(DE, "§1.2.3"), tests=["ENCAP-PHONE-*"]),
    Req("DIS-07", "驾驶相关注视不报警", "后视镜/仪表注视不发分心报警，只联动 ADAS 灵敏度", "功能",
        k(DE, "§1.4 车辆响应表"), tests=["ENCAP-LD-driving-*", "ENCAP-VATS-driving-*"]),
    Req("DIS-08", "ADAS 灵敏度联动", "连续离开道路 ≥0.8 s（≤1 s）置位 FSS 请求；短扫视在连续注视道路 ≥1 s 后、分心/疲劳状态在结束 ≥2 s 后复位", "功能",
        k(DE, "§1.4.3 表", "§1.4.3.1") + k(SD, "§4.2.2.1"), da=["DA-17"],
        tests=["BC-FSS-*", "ENCAP-LD-driving-*"]),
    Req("DIS-09", "视觉预警（附加策略）", "非驾驶单次注视 ≥2.0 s 或累计 ≥7 s 时先给视觉预警", "HMI",
        k(ADDW, P1 + "§3.3.2.5–3.3.2.6", P1 + "§3.4.1.2"), da=["DA-05", "DA-06"], tests=["BC-PRE-*", "FA-pre=2.5s"]),
    # ------------------------------------------------------------------ drowsiness
    Req("DRW-01", "PERCLOS", "60 s 窗内眼睑闭合度 ≥0.8 的帧占比；眼部不可见的帧不计入，有效帧不足一半时不输出", "功能",
        k(DDAW, P2 + "§5.2.1(a)"), da=["DA-10", "DA-11"]),
    Req("DRW-02", "疲劳分级", "L1 信息：PERCLOS ≥0.12；L2 报警：PERCLOS ≥0.20（假设对应 KSS ≥8，须按 DDAW Part 2 §6.1 标定）；带滞回", "功能",
        k(DDAW, P1 + "§3.3.1", P1 + "附录", P2 + "§6.1") + k(DE, "§1.3.2", "§1.4.2"), da=["DA-12", "DA-13", "DA-15"],
        tests=["BC-PERCLOS-*", "SYS-ARBITRATION", "DD-SYN"]),
    Req("DRW-03", "疲劳功能工作车速", "疲劳（PERCLOS）报警在 ≥50 km/h 工作（覆盖 DDAW 的 >70 km/h 激活与 65–130 km/h 范围）", "功能",
        k(DE, "§1.1 一般要求（车速）") + k(DDAW, P1 + "§3.1.4–3.1.6"), tests=["BC-DROWSY-SPD-*"]),
    Req("DRW-04", "学习期", "PERCLOS 窗满 60 s 后才输出疲劳等级（远小于 10 min / 5 min 上限）", "功能",
        k(DE, "§1.1 一般要求（车速）") + k(DDAW, P1 + "§3.1.4–3.1.6", P2 + "§8.2"), da=["DA-11"]),
    Req("DRW-05", "微睡眠", "车速 ≥20 km/h 时闭眼 ≥1.2 s 报警（声+触+视），睁眼后保持 2 s；申报 1.3 s", "功能",
        k(DE, "§1.3.3") + k(SD, "§4.1.2.2–4.1.2.3"), da=["DA-14"], tests=["ENCAP-MS", "BC-MS-*", "IVASA-FAT-*"]),
    Req("DRW-06", "睡眠", "闭眼 ≥3.0 s 报警", "功能", k(DE, "§1.3.4"), tests=["ENCAP-SLEEP", "BC-SLEEP-*"]),
    Req("DRW-07", "无响应", "闭眼 ≥6 s 或分心报警后 3 s 内视线未回到道路 → 无响应，输出 EF 请求标志", "功能",
        k(DE, "§1.3.5", "§1.4.3.3") + k(SD, "§4.2.3.3"), tests=["ENCAP-UNRESP-*", "BC-UNRESP-*"]),
    Req("DRW-08", "睡眠类报警更紧急", "微睡眠/睡眠/无响应报警使用更高紧急等级，与分心、疲劳报警区分", "HMI", k(DE, "§1.4.2")),
    Req("DRW-09", "基于车辆信号的 DDAW 主判据", "DDAW 型式认证要求分析车辆系统（转向、车道位置）；摄像头指标只作辅助参数", "功能",
        k(DDAW, P1 + "§3.3.2") + k(GSR, "Art. 3(5)"),
        na="本项目只做摄像头侧逻辑，没有转向/车道数据；需与底盘、车道保持数据融合，并按 DDAW Part 2 做真人验证"),
    Req("DRW-10", "DDAW 验证方法", "按 Part 2 实现 TP/FN 判定（KSS 每 5 min、前后 KSS ≥7、跨阈值规则）和有效性判据（>40 % 或下界 >20 %）", "试验",
        k(DDAW, P2 + "§3.1/§3.4", P2 + "§5.1.2", P2 + "§5.1.4–5.1.5", P2 + "§8.1", P2 + "§8.2"), da=["DA-21"], tests=["DD-SYN"]),
    # ------------------------------------------------------------------ availability
    Req("AVL-01", "永久故障与无光照遮挡", "相机故障或测不到光持续 1.0 s → 恒定故障指示，故障期间保持，消失 2 s 后清除", "功能",
        k(ADDW, P1 + "§3.5.1.1–3.5.1.2", P1 + "§3.5.1.3", P1 + "§3.5.1.4") + k(DDAW, P1 + "§3.5.1/§3.5.3"), da=["DA-16"],
        tests=["SYS-FAILURE-NOLIGHT"]),
    Req("AVL-02", "运行前自检", "上电自检完成后才进入运行", "功能", k(ADDW, P1 + "§3.5.1.1–3.5.1.2"),
        na="属于 ECU 诊断软件，需在 HIL / 实车上验证"),
    Req("AVL-03", "暂时不可用提示", "人脸丢失或眼部不可见持续 5 s（≤10 s）→ 视觉提示（持续）+ 每行程一次声音提示", "HMI",
        k(ADDW, P1 + "§3.5.2.1–3.5.2.2") + k(DE, "§1.1.1.2 Inform if not functional") + k(SD, "§4.3"), da=["DA-16"],
        tests=["ENCAP-NF-hand", "ENCAP-OCC-sunglasses-inform", "BC-INFORM-*"]),
    Req("AVL-04", "降级模式", "眼部不可见但头部可跟踪：用头部姿态继续检测 owl 型分心，lizard 型与眼睑指标不可用并提示", "功能",
        k(DE, "定义 Degraded / Non-functional"),
        tests=["ENCAP-OCC-sunglasses-LD-owl", "ENCAP-OCC-sunglasses-LD-lizard", "ENCAP-OCC-sunglasses-MS"]),
    Req("AVL-05", "人群与遮挡鲁棒性", "年龄、身材、肤色、眼睑开度、眼镜、口罩、胡须等条件下感知正常", "性能",
        k(DE, "§1.1.1.1–1.1.1.2") + k(SD, "§4/附录 A") + k(ADDW, P2 + "§1.2.2–1.3.1"),
        na="感知算法与数据集问题，逻辑层只消费标签；需按噪声变量做数据集评估和场地抽查"),
    Req("AVL-06", "定期技术检验", "故障信号状态与电子接口可供检查", "功能", k(ADDW, P1 + "§3.6"), na="诊断接口不在本项目范围"),
    # ------------------------------------------------------------------ HMI
    Req("HMI-01", "报警构成与起点", "分心/睡眠类报警同时有视觉（告知）和声音（警示）；报警起点 = 声音开始", "HMI",
        k(ADDW, P1 + "§3.4.1.1") + k(DE, "§1.4.1", "§1.4.2") + k(DDAW, P1 + "§3.4.1.1/§3.4.3.2"),
        tests=["ADDW-P2-*", "ENCAP-LD-nondriving-*"]),
    Req("HMI-02", "逐级升级", "视觉预警（2.0 s）→ 声音（3.3 s 或累计 10 s）→ 触觉（声音后 1.5 s 仍未消除；60 s 内再次报警直接触觉）", "HMI",
        k(ADDW, P1 + "§3.4.1.1", P1 + "§3.4.1.2"), da=["DA-05", "DA-07", "DA-08"]),
    Req("HMI-03", "终止条件", "触发条件消失即停止（长注视：回到道路 >0.3 s；VATS：连续注视道路 ≥2 s）；声音至少 1.0 s；视觉保持 2 s", "HMI",
        k(ADDW, P1 + "§3.4.1.1", P1 + "§3.4.2–3.4.4"), da=["DA-09"]),
    Req("HMI-04", "报警物理参数", "声音 200–8000 Hz、50–90 dB，语音与文字一致，视觉昼夜可见，触觉可感知", "HMI",
        k(ADDW, P1 + "§3.4.2–3.4.4") + k(DDAW, P1 + "§3.4.1.1/§3.4.3.2"),
        na="HMI 硬件与声学实现，需台架测量（频谱、声压级）和主观评价"),
    Req("HMI-05", "提示仲裁", "同一时刻只有一个 DMS 提示源：无响应 > 睡眠 > 微睡眠 > 分心 > 疲劳报警 > 视觉预警 > 信息；被压住的疲劳报警延后补发", "HMI",
        k(GSR, "Art. 6(3)"), tests=["SYS-ARBITRATION"]),
    Req("HMI-06", "ADAS 紧急报警时抑制", "FCW/AEB 等报警期间去掉 DMS 的声音和触觉（保留视觉），结束后恢复", "HMI",
        k(ADDW, P1 + "§3.1.5"), tests=["SYS-ADAS-SUPPRESS"]),
    Req("HMI-07", "确认后抑制（可选）", "Euro NCAP 允许驾驶员确认首次报警后抑制瞬态报警；本设计不采用", "HMI",
        k(DE, "§1.4.1"), na="设计选择不实现：避免驾驶员习惯性确认后失去保护；如采用需另加测试"),
    # ------------------------------------------------------------------ test procedures
    Req("TST-01", "ADDW 抽查程序", "实现 Part 2：区域 (a)–(n)、两档车速、60 s/15 s 前置、保持至期望时间 +3 s、4.0/6.5 s 假阴性、复测与通过准则", "试验",
        k(ADDW, P2 + "§1.2.2–1.3.1", P2 + "§1.4.2", P2 + "§1.5.1", P2 + "§1.6", P2 + "§2.3.1/§2.3.5/§2.3.9", P2 + "§2.3.8",
          P2 + "§3.1", P2 + "§3.2", P2 + "§4–§6"), tests=["ADDW-P2-*"]),
    Req("TST-02", "Euro NCAP SD-202 程序", "T0 = T_away − 4 s、20–80 km/h、注视 ≤5 s、容差 0.2 / 2.0 s、闭眼 4 / 7 s、13 / 14 s 无响应、10 s 遮挡、遮挡配件", "试验",
        k(SD, "§1/§1.1", "§3.2.1", "§3.2.3.2", "§4/附录 A", "§4.1", "§4.1.1.1", "§4.1.1.2", "§4.1.2.2–4.1.2.3",
          "§4.2.3.3", "§4.3") + k(DE, "§1.2.1"), tests=["ENCAP-*"]),
    Req("TST-03", "FSS 与 AEB 联动抽查", "在 CCR 场景中验证分心时 FCW/AEB 触发提前 ≥200 ms", "试验",
        k(SD, "§4.2.2.1") + k(DE, "§1.4.3.1"),
        na="需要闭环 AEB 模型；本项目只验证 FSS 请求的时延（DIS-08），ΔTTC 留待与 P5 的 AEB 闭环联调"),
    Req("TST-04", "IVASA 座舱 DMS 试验", "三个分心点各注视 5 s、闭眼 5 s，各 3 次，阈值内报警", "试验",
        k(IVA, "§3.1.2/§3.2", "§4.1.1", "§4.1.2"), tests=["IVASA-*"]),
    Req("TST-05", "NHTSA EGDS 验收", "24 名合成受试者（年龄性别配比），三条扫视准则各需 ≥21/24", "试验",
        k(NH, "VI.A.4–VI.A.5", "IV.A.9–IV.A.10", "VI.E.14(a)", "VI.E.14(b)", "VI.E.14(c)"), da=["DA-20"], tests=["NH-*"]),
    # ------------------------------------------------------------------ HMI task design
    Req("HMT-01", "行驶中视觉-手动任务的验收与锁止", "行驶中可用的视觉-手动任务须满足 NHTSA 准则，否则行驶中锁止", "HMI 任务",
        k(NH, "V.G", "VI.E.14(a)", "VI.E.14(b)", "VI.E.14(c)", "VI.G.17"), da=["DA-20"],
        tests=["NH-touch_address", "NH-voice_address", "NH-touch_favourite"]),
    Req("HMT-02", "导航目的地语音输入", "导航到地址/POI/收藏须提供直接语音输入，并保留替代控制和专用语音按键", "HMI 任务",
        k(DE, "§2.3.3"), tests=["NH-voice_address"]),
    Req("HMT-03", "固定锁止项", "行驶中锁止用于通讯/上网的手动文本输入、视频等", "HMI 任务", k(NH, "V.F", "IV.A.6"),
        na="属座舱 HMI 功能清单，本项目只评估导航任务；需在 HMI 规范中逐项列出并测试"),
    # ------------------------------------------------------------------ documentation
    Req("DOC-01", "型式认证文件包", "局限说明、指标、区域定义、注视点、HMI 升级策略、验证证据；非 KSS 指标与 KSS 8 的对应关系", "文档",
        k(ADDW, P3 + "§1–§3") + k(DDAW, P2 + "§6.1"),
        na="本项目的 需求规格.md 和 report.md 只能作为草稿结构；正式文件包需真实验证数据"),
]


def by_id() -> dict[str, Req]:
    return {r.rid: r for r in REQS}
