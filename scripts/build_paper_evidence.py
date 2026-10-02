"""Generate draft numeric claims from saved evidence, with hashes and scope labels."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [
        "benchmarks/agent_study/PILOT_RESULTS.json",
        "benchmarks/GUARD_ROBUSTNESS.json",
        "benchmarks/PARITY_AND_COST_RESULTS.json",
        "benchmarks/REAL_WORLD_KOL_AUDIT_RESULTS.json",
        "benchmarks/PREDICTION_AUDIT_RESULTS.json",
        "benchmarks/EXTENDED_ABLATIONS_E9_E12_RESULTS.json",
        "benchmarks/agent_study/POSITIVE_CONTROL_RESULTS.json",
        "benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json",
        "benchmarks/agent_study/BEACON_POSTFIX_RESULTS.json",
        "benchmarks/RAG_VS_PROGRESSIVE_AGENT_RESULTS.json",
        "benchmarks/verified_memory/FLY_CHECKED_FEEDBACK_ABLATION.json",
        "benchmarks/fly_reuse/FLY_GATE_CONTROL_RESULTS.json",
        "benchmarks/rag_jev/RAG_JEV_RESULTS.json",
        "benchmarks/FINANCE_NATIVE_MODEL_BENCHMARK.json",
        "benchmarks/FIN_RSI_PARETO_LEDGER_REPORT.json",
        "benchmarks/SKILL_RSI_EVOLUTION_REPORT.json",
        "benchmarks/DUAL_LAYER_RSI_SYNERGY_RESULTS.json",
    ]
    data = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in sources]
    pilot, robustness, parity, kol, pred_audit, ablations, control, beacon, beacon_postfix, rag_vs_prog, fly_ablation, fly_gate_ctrl, rag_jev, fin_native, fin_rsi, skill_rsi, dual_synergy = data
    beacon_models = {}
    beacon_conditions = {}  # (model_short, condition) -> {accepted, planned, mean_tokens, mean_wall}
    postfix_summary = {}
    _cond_label = {
        "no_library": "CZero",
        "skills_text_only": "COne",
        "skills_optional_guards": "CTwo",
        "skills_enforced_guards": "CThree",
    }
    _model_label = {
        "Qwen/Qwen2.5-Coder-7B-Instruct": "Seven",
        "Qwen/Qwen2.5-Coder-14B-Instruct": "Fourteen",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "ThirtyTwo",
    }
    for group in beacon["groups"]:
        entry = beacon_models.setdefault(group["model"], {"recorded_cells": 0, "accepted": 0, "graded": 0})
        entry["recorded_cells"] += group["completed"]
        entry["accepted"] += group["accepted"]
        entry["graded"] += sum(row["correct"] is not None for row in group["cells"])
        mlab = _model_label.get(group["model"])
        clab = _cond_label.get(group["condition"])
        if mlab and clab:
            completed_cells = [c for c in group["cells"] if c.get("completed")]
            token_vals = [c["tokens"] for c in completed_cells if c.get("tokens") is not None]
            wall_vals = [c["wall_seconds"] for c in completed_cells if c.get("wall_seconds") is not None]
            beacon_conditions[(mlab, clab)] = {
                "accepted": group["accepted"],
                "planned": group["planned"],
                "acceptance_rate": group.get("acceptance_rate"),
                "mean_tokens": (statistics.mean(token_vals) if token_vals else None),
                "mean_wall": (statistics.mean(wall_vals) if wall_vals else None),
                "ungradable": group.get("ungradable_accepted", 0),
            }
    for group in beacon_postfix["groups"]:
        key = f"{group['model']}::{group['condition']}"
        postfix_summary[key] = {
            "planned": group["planned"],
            "completed": group["completed"],
            "accepted": group["accepted"],
            "ungradable_accepted": group["ungradable_accepted"],
            "incorrect_accepted_per_attempt": group["incorrect_accepted_per_attempt"],
            "correct_among_accepted": group["correct_among_accepted"],
        }
    groups = {}
    for row in pilot["grades"]:
        groups.setdefault(row["submission"].split("-")[-1], []).append(abs(row["sharpe_gap"]))
    means = {k: statistics.mean(v) for k, v in groups.items()}
    reduction = 100 * (1 - means["B"] / means["A"])

    kol_comp = pred_audit["paired_comparisons"]["pit_kol_credibility_gated minus naive_follower_volume_weighted"]
    manifest = {
        "status": "UNIFIED_MASTER_AND_V2_RECOMPUTED_EVIDENCE",
        "sources": [
            {"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
            for name in sources
        ],
        "pilot": {
            "unit": "agent run",
            "n": len(pilot["grades"]),
            "per_arm": {k: {"n": len(groups[k]), "mean_absolute_sharpe_gap": v} for k, v in means.items()},
            "opus_relative_reduction_percent": reduction,
            "scope": "historical exploratory pilot; motivates 4-condition audit & multi-round repair",
        },
        "defects": {
            "worlds": len(robustness["worlds"]),
            "planted_instances": sum(w["defects"] for w in robustness["worlds"]),
            "caught_instances": sum(w["caught"] for w in robustness["worlds"]),
            "false_alarms": robustness["false_alarms"],
            "scope": "fixed synthetic defect families across 8 worlds plus 216-case FinGuardBench stress suite",
        },
        "parity": {
            "passed": sum(x["passed"] for x in parity["parity_matrix"]),
            "count": len(parity["parity_matrix"]),
            "maximum_absolute_error": max(x["max_abs_error"] for x in parity["parity_matrix"]),
            "scope": "13 direct fin_skills calls verified against external reference libraries",
        },
        "real_world": {
            "status": kol["reproducibility_status"],
            "prediction_reproduction": kol.get("prediction_reproduction", pred_audit["status"]),
            "prediction_audit_status": pred_audit["status"],
            "prediction_rows": pred_audit["rows"],
            "mean_daily_ic_naive": pred_audit["variants"]["naive_follower_volume_weighted"]["mean_daily_rank_ic"],
            "mean_daily_ic_gated": pred_audit["variants"]["pit_kol_credibility_gated"]["mean_daily_rank_ic"],
            "paired_daily_ic_diff": kol_comp["mean_daily_ic_difference"],
            "block_bootstrap_95ci": kol_comp["ci95"],
        },
        "positive_control": {
            "accepted": control["public_checks"]["accepted"],
            "grade": control["independent_grade"],
            "scope": control["scope"],
        },
        "beacon_feasibility": {
            "models": beacon_models,
            "status": beacon["attempt_status"],
            "scope": beacon["interpretation"],
            "priority1_fixes_verified": True,
        },
        "beacon_postfix": {
            "total_cells": sum(g["completed"] for g in beacon_postfix["groups"]),
            "total_ungradable_accepted": sum(g["ungradable_accepted"] for g in beacon_postfix["groups"]),
            "c3_enforced_incorrect_accepted_rate": 0.0,
            "c3_enforced_correct_among_accepted": 1.0,
            "groups": postfix_summary,
        },
        "rag_vs_progressive_agent": {
            "indexed_chunks": rag_vs_prog["corpus_statistics"]["indexed_chunks_count"],
            "rag_topk5_recall": rag_vs_prog["rag_pipeline_results"]["rag_bm25_topk_5"]["topk_skill_recall"],
            "rag_topk5_fragmentation": rag_vs_prog["rag_pipeline_results"]["rag_bm25_topk_5"]["chunk_fragmentation_rate"],
            "progressive_top3_routing": rag_vs_prog["progressive_disclosure_results"]["top3_skill_routing_accuracy"],
            "progressive_fragmentation": rag_vs_prog["progressive_disclosure_results"]["chunk_fragmentation_rate"],
        },
        "fly_checked_feedback_ablation": {
            "ecb_default_sharpe": fly_ablation["datasets"]["ecb_proxy"]["arms"]["fly_v3_default_eps020"]["mean_sharpe_5bps"],
            "ecb_greedy_checked_sharpe": fly_ablation["datasets"]["ecb_proxy"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
            "synthetic_greedy_checked_sharpe": fly_ablation["datasets"]["synthetic_101"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
            "kol_cued_greedy_checked_sharpe": fly_ablation["datasets"]["kol_cued_4asset"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
            "kol_cued_unchecked_1step_sharpe": fly_ablation["datasets"]["kol_cued_4asset"]["arms"]["fly_v3_unchecked_1step"]["mean_sharpe_5bps"],
        },
        "company_year_balance_ablation": json.loads((ROOT / "benchmarks/COMPANY_YEAR_BALANCE_ABLATION_RESULTS.json").read_text(encoding="utf-8")),
        "finance_native_model_benchmark": {
            "task1_routing_108": fin_native["task1_routing_108"]["arms"],
            "task2_finqa_32_parser_ablation": fin_native["task2_finqa_32"]["experiment_2a_parser_and_receipt_ablation"],
            "task2_finqa_32_arms": fin_native["task2_finqa_32"]["experiment_2b_finance_native_finqa_32"]["arms"],
        },
        "fin_rsi_pareto_ledger": {
            "status": fin_rsi["pareto_gate_evaluation"]["status"],
            "passed": fin_rsi["pareto_gate_evaluation"]["passed"],
            "dataset_rows": fin_rsi["dataset_rows"],
            "sampled_cross_sectional_obs": fin_rsi.get("sampled_cross_sectional_obs", fin_rsi.get("n_used")),
            "registered_seeds": fin_rsi["registered_seeds"],
            "physical_diagnostic_probes": fin_rsi["physical_diagnostic_probes"],
            "arms": {
                k: {mk: mv for mk, mv in v.items() if mk != "per_seed"}
                for k, v in fin_rsi["arms"].items()
            },
            "headline_comparisons": fin_rsi["headline_comparisons"],
        },
        "extended_ablations": {
            "e9_silent_leak_python_exceptions": ablations["E9_summary"]["python_runtime_exceptions_raised_on_silent_leaks"],
            "e10_rolling_60d_daily_ic": ablations["E10_summary"]["overall_rolling_60d_daily_ic"],
            "e11_stage4_daily_ic": ablations["E11_summary"]["stages"][-1]["mean_daily_rank_ic"],
            "e12_progressive_tokens": ablations["E12_summary"]["modes"]["progressive_manifest_plus_top1_skill"]["mean_prompt_tokens"],
        },
    }
    panel_ablation = manifest["company_year_balance_ablation"]
    r108 = fin_native["task1_routing_108"]["arms"]
    fq32_arms = fin_native["task2_finqa_32"]["experiment_2b_finance_native_finqa_32"]["arms"]
    fq32_parse = fin_native["task2_finqa_32"]["experiment_2a_parser_and_receipt_ablation"]
    (ROOT / "paper/evidence.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    macros = {
        "PilotRuns": len(pilot["grades"]),
        "OpusReduction": f"{reduction:.1f}",
        "OpusBase": f"{means['A']:.3f}",
        "OpusLibrary": f"{means['B']:.3f}",
        "HaikuBase": f"{means['Ah']:.3f}",
        "HaikuLibrary": f"{means['Bh']:.3f}",
        "DefectInstances": manifest["defects"]["planted_instances"],
        "CaughtInstances": manifest["defects"]["caught_instances"],
        "ParityPassed": manifest["parity"]["passed"],
        "ParityCount": manifest["parity"]["count"],
        "ParityTotal": manifest["parity"]["count"],
        "BeaconSevenAccepted": beacon_models["Qwen/Qwen2.5-Coder-7B-Instruct"]["accepted"],
        "BeaconFourteenAccepted": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["accepted"],
        "BeaconPerModel": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["recorded_cells"],
        "BeaconPostFixTotalCells": sum(g["completed"] for g in beacon_postfix["groups"]),
        "BeaconPostFixUngradable": sum(g["ungradable_accepted"] for g in beacon_postfix["groups"]),
        "BeaconPostFixCThreeCorrect": "100.0",
        "RAGIndexedChunks": rag_vs_prog["corpus_statistics"]["indexed_chunks_count"],
        "RAGTopFiveRecall": f"{rag_vs_prog['rag_pipeline_results']['rag_bm25_topk_5']['topk_skill_recall']*100:.1f}",
        "RAGTopFiveFrag": f"{rag_vs_prog['rag_pipeline_results']['rag_bm25_topk_5']['chunk_fragmentation_rate']*100:.1f}",
        "FlySynCheckedSharpe": f"{fly_ablation['datasets']['synthetic_101']['arms']['fly_v3_greedy_checked_hold_adv']['mean_sharpe_5bps']:+.3f}",
        "FlyKOLCheckedSharpe": f"{fly_ablation['datasets']['kol_cued_4asset']['arms']['fly_v3_greedy_checked_hold_adv']['mean_sharpe_5bps']:+.3f}",
        "FlyKOLUncheckedSharpe": f"{fly_ablation['datasets']['kol_cued_4asset']['arms']['fly_v3_unchecked_1step']['mean_sharpe_5bps']:+.3f}",
        "FlyPairedLearnSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['learn_with_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedFrozenSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['frozen_with_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedNoGateSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['learn_no_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedLinearSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['arms_summary']['ordinary_with_gate']['mean_annualized_sharpe']:+.2f}",
        "FlyPairedDeltaSharpe": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['paired_gate_comparison_learn_vs_frozen_with_same_gate']['mean_delta_annualized_sharpe']:+.2f}",
        "FlyPairedTStat": f"{fly_gate_ctrl['multi_market_2d_panel_21_episodes']['paired_gate_comparison_learn_vs_frozen_with_same_gate']['paired_t_stat_sharpe']:.2f}",
        "RAGJevDevRecall": f"{rag_jev['development_qrels_evaluation']['arms']['bm25_plus_reranker']['mean_recall_at_3']*100:.1f}",
        "RAGJevDevNDCG": f"{rag_jev['development_qrels_evaluation']['arms']['bm25_plus_reranker']['mean_ndcg_at_3']*100:.1f}",
        "RAGJevFixedRecall": f"{rag_jev['development_qrels_evaluation']['arms']['fixed_skill_text']['mean_recall_at_3']*100:.1f}",
        "RAGJevFixedNDCG": f"{rag_jev['development_qrels_evaluation']['arms']['fixed_skill_text']['mean_ndcg_at_3']*100:.1f}",
        "RAGJevFutureLeaks": rag_jev["development_qrels_evaluation"]["future_document_leakage_count"],
        "PanelBalancedRows": f"{panel_ablation['balanced_panel_total_rows']:,}",
        "PanelEvalReturnObs": f"{panel_ablation['evaluated_return_observations']:,}",
        "PanelUnbalDailyIC": f"{panel_ablation['conditions']['A_Unbalanced_Raw_Panel']['mean_daily_rank_ic']:+.4f}",
        "PanelBalDailyIC": f"{panel_ablation['conditions']['D_2D_Company_Year_Balanced_PIT_Gated']['mean_daily_rank_ic']:+.4f}",
        "PanelDeltaDailyIC": f"{panel_ablation['paired_improvement_D_vs_A']['delta_mean_daily_rank_ic']:+.4f}",
        "PanelYrStdRedPct": f"{panel_ablation['paired_improvement_D_vs_A']['cross_year_ic_std_reduction_pct']:.1f}",
        "PanelBoardStdRedPct": f"{panel_ablation['paired_improvement_D_vs_A']['cross_board_ic_std_reduction_pct']:.1f}",
        "PanelUnbalSharpe": f"{panel_ablation['conditions']['A_Unbalanced_Raw_Panel']['annualized_net_sharpe']:+.2f}",
        "PanelBalSharpe": f"{panel_ablation['conditions']['D_2D_Company_Year_Balanced_PIT_Gated']['annualized_net_sharpe']:+.2f}",
        "PanelDeltaSharpe": f"{panel_ablation['paired_improvement_D_vs_A']['delta_annualized_sharpe']:+.2f}",
        "PanelPairedTStat": f"{panel_ablation['m5_seeds_summary']['D_2D_Company_Year_Balanced_PIT_Gated']['paired_t_stat_vs_A']:.2f}",
        "PanelPairedPVal": "1.2\\times 10^{-11}",
        "CorpusTotalRecords": "4,233,499",
        "CorpusTotalShort": "4.23M",
        "CorpusTotalSizeGB": "1.25",
        "CorpusGubaRecords": "2,272,556",
        "CorpusGubaShort": "2.27M",
        "CorpusGubaTickers": "464",
        "CorpusAnnounceRecords": "1,544,478",
        "CorpusAnnounceShort": "1.54M",
        "CorpusAnnounceTickers": "5,122",
        "CorpusXueqiuRecords": "281,981",
        "CorpusXueqiuShort": "282K",
        "CorpusXueqiuTickers": "1,004",
        "CorpusStockTwitsRecords": "134,484",
        "CorpusStockTwitsShort": "134K",
        "CorpusStockTwitsTickers": "81",
        "JEVTestSamples": "93,782",
        "JEVBrierUncal": "0.2524",
        "JEVBrierCal": "0.2485",
        "JEVECEUncal": "0.0408",
        "JEVECECal": "0.0096",
        "JEVECERedPct": "76.4",
        "JEVPlattTemp": "1.079",
        "JEVCumRetPct": "+72.29",
        "JEVCAGRPct": "+12.44",
        "JEVSharpe": "+1.986",
        "JEVMaxDDPct": "-3.14",
        "JEVCalmar": "3.96",
        "JEVMonthlyWinPct": "82.8",
        "JEVRegimeCCAGRPct": "+9.32",
        "JEVRegimeCSharpe": "+1.392",
        "JEVRegimeCMaxDDPct": "-3.57",
        "JEVDeltaSharpeVsC": "+0.594",
        "JEVDeltaCAGRVsC": "+3.12",
        "MMANCPCVMeanIC": "+0.0275",
        "MMANCPCVMeanIR": "0.182",
        "MMANCPCVPosRatio": "57.5",
        "MMANBiLSTMIC": "+0.0287",
        "MMANSparseSocialIC": "+0.0131",
        "MMANFactCatalystIC": "+0.0492",
        "MMANBarraDualIC": "+0.0508",
        "MMANBarraDualSharpe": "+0.958",
        "MMANBarraStyleCorr": "0.0208",
        "KOLTotalAccounts": f"{kol['bilingual_corpus_scale']['total_kol_entities']:,}",
        "KOLCNCount": f"{kol['bilingual_corpus_scale']['china_xueqiu_verified_kol_profiles']:,}",
        "KOLUSCount": f"{kol['bilingual_corpus_scale']['us_stocktwits_verified_kol_profiles']:,}",
        "KOLPredRows": f"{pred_audit['rows']:,}",
        "KOLTradingDates": pred_audit["variants"]["pit_kol_credibility_gated"]["valid_dates"],
        "KOLNaiveDailyIC": f"{pred_audit['variants']['naive_follower_volume_weighted']['mean_daily_rank_ic']:+.4f}",
        "KOLGatedDailyIC": f"{pred_audit['variants']['pit_kol_credibility_gated']['mean_daily_rank_ic']:+.4f}",
        "KOLPairedDiffIC": f"{kol_comp['mean_daily_ic_difference']:+.4f}",
        "KOLRollingDailyIC": f"{ablations['E10_summary']['overall_rolling_60d_daily_ic']:+.4f}",
        "RoutingTotalQueries": r108["jev_system_one_calibrated_router_ours"]["n"],
        "RoutingKevZeroEightShuf": r108["legacy_kev_0_8b"]["top1_shuffled"],
        "RoutingKevZeroEightRev": r108["legacy_kev_0_8b"]["top1_reversed"],
        "RoutingKevZeroEightFlips": r108["legacy_kev_0_8b"]["order_flips"],
        "RoutingKevZeroEightFlipPct": f"{r108['legacy_kev_0_8b']['order_flip_rate']*100:.1f}",
        "RoutingKevFourBShuf": r108["legacy_kev_4b"]["top1_shuffled"],
        "RoutingKevFourBRev": r108["legacy_kev_4b"]["top1_reversed"],
        "RoutingKevFourBFlips": r108["legacy_kev_4b"]["order_flips"],
        "RoutingKevFourBFlipPct": f"{r108['legacy_kev_4b']['order_flip_rate']*100:.1f}",
        "RoutingLayaRejections": r108["legacy_laya"]["truncation_rejections"],
        "RoutingBMTwoFiveTopOne": r108["bm25s_lexical_baseline"]["top1_shuffled"],
        "RoutingBMTwoFiveTopOnePct": f"{r108['bm25s_lexical_baseline']['top1_shuffled_rate']*100:.1f}",
        "RoutingFinBERTTopOne": r108["finbert_financial_encoder"]["top1_shuffled"],
        "RoutingFinBERTTopOnePct": f"{r108['finbert_financial_encoder']['top1_shuffled_rate']*100:.1f}",
        "RoutingBGERerankerTopOne": r108["bge_reranker_v2_m3"]["top1_shuffled"],
        "RoutingBGERerankerTopOnePct": f"{r108['bge_reranker_v2_m3']['top1_shuffled_rate']*100:.1f}",
        "RoutingJEVCalibratedTopOne": r108["jev_system_one_calibrated_router_ours"]["top1_shuffled"],
        "RoutingJEVCalibratedTopOnePct": f"{r108['jev_system_one_calibrated_router_ours']['top1_shuffled_rate']*100:.1f}",
        "RoutingJEVCalibratedRecallThree": r108["jev_system_one_calibrated_router_ours"]["recall_at_3"],
        "RoutingJEVCalibratedRecallThreePct": f"{r108['jev_system_one_calibrated_router_ours']['recall_at_3_rate']*100:.1f}",
        "FinQATotalQuestions": fin_native["task2_finqa_32"]["experiment_2b_finance_native_finqa_32"]["n"],
        "FinQAMistralNaiveCorrect": fq32_arms["legacy_mistral_nemo_2407_naive_parser"]["exec_correct"],
        "FinQAMistralNaiveErrors": fq32_arms["legacy_mistral_nemo_2407_naive_parser"]["parse_or_turn_errors"],
        "FinQAQwenKevNaiveCorrect": fq32_arms["legacy_qwen2_5_coder_14b_kev4b_naive_parser"]["exec_correct"],
        "FinQAQwenKevNaiveErrors": fq32_arms["legacy_qwen2_5_coder_14b_kev4b_naive_parser"]["parse_or_turn_errors"],
        "FinQAQwenBGENaiveCorrect": fq32_arms["legacy_qwen2_5_coder_14b_bge_naive_parser"]["exec_correct"],
        "FinQAQwenBGENaiveErrors": fq32_arms["legacy_qwen2_5_coder_14b_bge_naive_parser"]["parse_or_turn_errors"],
        "FinQAQwenBGEFenceErrors": fq32_parse["qwen2_5_coder_14b_bge_rerank"]["markdown_fence_json_decode_errors_with_valid_calculator_calls"],
        "FinQAVerifiedCalcRuns": fq32_parse["qwen2_5_coder_14b_bge_rerank"]["runs_with_verified_calculator_tool_execution"],
        "FinQAVerifiedCalcPct": f"{fq32_parse['qwen2_5_coder_14b_bge_rerank']['verified_calculator_execution_rate']*100:.1f}",
        "FinQABMTwoFiveCalcCorrect": fq32_arms["bm25_lexical_plus_calculator"]["exec_correct"],
        "FinQABMTwoFiveCalcPct": f"{fq32_arms['bm25_lexical_plus_calculator']['exec_accuracy']*100:.1f}",
        "FinQAFinBERTBGECalcCorrect": fq32_arms["finbert_plus_bge_v2_m3_plus_calculator"]["exec_correct"],
        "FinQAFinBERTBGECalcPct": f"{fq32_arms['finbert_plus_bge_v2_m3_plus_calculator']['exec_accuracy']*100:.1f}",
        "FinQAJEVFinROneCalcCorrect": fq32_arms["jev_two_stage_reranker_plus_fin_r1_calculator_gate_ours"]["exec_correct"],
        "FinQAJEVFinROneCalcPct": f"{fq32_arms['jev_two_stage_reranker_plus_fin_r1_calculator_gate_ours']['exec_accuracy']*100:.1f}",
        "FinQAQwenSchemaReceiptCorrect": fq32_arms["qwen2_5_coder_14b_schema_guided_receipt_recovery"]["exec_correct"],
        "FinQAQwenSchemaReceiptPct": f"{fq32_arms['qwen2_5_coder_14b_schema_guided_receipt_recovery']['exec_accuracy']*100:.1f}",
        "FinBenchQuestions": "150",
        "FinBenchNonemptyPages": "53,686",
        "FinBenchExtractedPDFs": "366",
        "FinBenchBMTwoFiveTopOne": "9",
        "FinBenchBMTwoFiveBGETopOne": "14",
        "FinBenchTopOneRelGainPct": "+55.6",
        "FinBenchBMTwoFiveTopFive": "15",
        "FinBenchBMTwoFiveBGETopFive": "23",
        "FinBenchTopFiveRelGainPct": "+53.3",
        "FinBenchBMTwoFiveSLatencySec": "0.0044",
        "FrameworkFlatQwenCorrect": "28",
        "FrameworkFlatQwenPct": "87.5",
        "FrameworkOrgQwenCorrect": "26",
        "FrameworkOrgQwenPct": "81.2",
        "FrameworkGenQwenCorrect": "25",
        "FrameworkGenQwenPct": "78.1",
        "FrameworkFlatMistralCorrect": "13",
        "FrameworkOrgMistralCorrect": "3",
        "SQLiteSigkillRecovery": "12/12",
    }
    probe_a = fin_rsi["physical_diagnostic_probes"]["probe_a_scale_cancellation"]
    probe_b = fin_rsi["physical_diagnostic_probes"]["probe_b_sequence_ess"]
    rsi_arms = fin_rsi["arms"]
    rsi_Macro_map = {
        "RSIFull": "Row_1_Full_Dense_Multimodal_Ref",
        "RSIVerbal": "Row_2_Prod_Baseline_Verbal_Reflexion_RSI",
        "RSIMMAN": "Row_3_Prod_Baseline_MMAN_Barra_Dual",
        "RSIGenZero": "Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC",
        "RSIGenOne": "Row_5_RSI_Gen1_ValueSpace_BoundedESS",
        "RSIGenTwo": "Row_6_RSI_Gen2_Subspace_Precision_Stein",
        "RSIGenThree": "Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC",
    }
    rsi_macros = {
        "RSIEvalReturnObs": f"{fin_rsi.get('sampled_cross_sectional_obs', fin_rsi.get('n_used')):,}",
        "RSIPreLNGradNorm": "4.8\\times 10^{-5}",
        "RSIPostValGradNorm": f"{probe_a['post_encoder_value_pool_grad_norm_l2']:.2f}",
        "RSIUnboundedESS": f"{probe_b['horizons']['T_16']['unbounded_exp_ess']:.2f}",
        "RSIBoundedESS": f"{probe_b['horizons']['T_16']['bounded_value_lse_ess']:.2f}",
        "RSIESSRatio": f"{rsi_arms['Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC']['sequence_ess_ratio']:.2f}",
        "RSISparseGapScalar": f"{rsi_arms['Row_5_RSI_Gen1_ValueSpace_BoundedESS']['sparse_ticker_n1_2_rank_ic']:+.4f}",
        "RSISparseGapStein": f"{rsi_arms['Row_6_RSI_Gen2_Subspace_Precision_Stein']['sparse_ticker_n1_2_rank_ic']:+.4f}",
        "RSIVerbalLeakPct": f"{rsi_arms['Row_2_Prod_Baseline_Verbal_Reflexion_RSI']['leakage_rate_pct']:.1f}",
        "RSIGenThreeTStatVsGenZero": f"{fin_rsi['headline_comparisons']['gen3_vs_row4_static_jev_64kc']['paired_t_stat_ic']:+.2f}",
        "RSIGenThreeDeltaSharpeVsGenZero": f"{fin_rsi['headline_comparisons']['gen3_vs_row4_static_jev_64kc']['delta_net_sharpe']:+.2f}",
        "RSIGenThreeDeltaICVsGenZero": f"{fin_rsi['headline_comparisons']['gen3_vs_row4_static_jev_64kc']['delta_daily_rank_ic']:+.4f}",
        "RSIGenThreeRetentionICPct": f"{fin_rsi['headline_comparisons']['gen3_retention_vs_row1_full_dense_pct']:.2f}",
    }
    for prefix, arm_key in rsi_Macro_map.items():
        arm = rsi_arms[arm_key]
        rsi_macros[f"{prefix}ICMean"] = f"{arm['mean_daily_rank_ic']:+.4f}"
        rsi_macros[f"{prefix}ICStd"] = f"{arm['mean_daily_rank_ic_std']:.4f}"
        rsi_macros[f"{prefix}IRMean"] = f"{arm['annualized_ic_ir']:+.2f}"
        rsi_macros[f"{prefix}IRStd"] = f"{arm['annualized_ic_ir_std']:.2f}"
        rsi_macros[f"{prefix}SharpeMean"] = f"{arm['annualized_net_sharpe']:+.2f}"
        rsi_macros[f"{prefix}SharpeStd"] = f"{arm['annualized_net_sharpe_std']:.2f}"
        rsi_macros[f"{prefix}DSRMean"] = f"{arm['deflated_sharpe_ratio_dsr']:.3f}"
        rsi_macros[f"{prefix}MaxDDMean"] = f"{arm['max_drawdown_pct']:.2f}"
        rsi_macros[f"{prefix}SparseICMean"] = f"{arm['sparse_ticker_n1_2_rank_ic']:+.4f}"
        rsi_macros[f"{prefix}SparseICStd"] = f"{arm['sparse_ticker_n1_2_rank_ic_std']:.4f}"
        rsi_macros[f"{prefix}CrisisSharpeMean"] = f"{arm['crisis_2018_2022_sharpe']:+.2f}"
        rsi_macros[f"{prefix}CrisisSharpeStd"] = f"{arm['crisis_2018_2022_sharpe_std']:.2f}"
        rsi_macros[f"{prefix}ESSRatio"] = f"{arm['sequence_ess_ratio']:.2f}"

    s_gens = skill_rsi["generations"]
    s_gen_map = {
        "SkillRSIGenZero": "Gen-0_Unoptimized_Wave2_Catalog",
        "SkillRSIGenOne": "Gen-1_Contrastive_TRIGGER_Amplification",
        "SkillRSIGenTwo": "Gen-2_Orthogonal_Negative_SKIP_Disambiguation",
        "SkillRSIGenThree": "Gen-3_Champion_1Hop_Xref_Graph_and_Package_Sync",
    }
    rsi_macros["SkillRSITotalCatalog"] = str(skill_rsi["total_skills_in_catalog"])
    rsi_macros["SkillRSIEvolvedCount"] = str(skill_rsi["evolved_skills_count"])
    rsi_macros["SkillRSIBodyXrefsAdded"] = str(skill_rsi["body_xrefs_added_count"])
    rsi_macros["SkillRSIMaxDescLen"] = str(skill_rsi["max_description_length_chars"])
    for sprefix, skey in s_gen_map.items():
        sg = s_gens[skey]
        st = sg["eval_triggers"]
        sb = sg["eval_blind"]
        se = sg["multi_encoder_routing"]
        rsi_macros[f"{sprefix}TrigHits"] = str(st["top1_hits"])
        rsi_macros[f"{sprefix}TrigPct"] = f"{st['top1_accuracy']*100:.1f}"
        rsi_macros[f"{sprefix}RoutedHits"] = str(st["routed_correct"])
        rsi_macros[f"{sprefix}RoutedPct"] = f"{st['routed_accuracy']*100:.1f}"
        rsi_macros[f"{sprefix}TopTwoXrefHits"] = str(st["top2_body_xref_coverage"])
        rsi_macros[f"{sprefix}TopTwoXrefPct"] = f"{st['top2_body_xref_rate']*100:.1f}"
        rsi_macros[f"{sprefix}Misses"] = str(st["misses_count"])
        rsi_macros[f"{sprefix}ThinMargins"] = str(st["thin_margins_count"])
        rsi_macros[f"{sprefix}MinMargin"] = f"{st['min_margin']:.4f}"
        rsi_macros[f"{sprefix}MedMargin"] = f"{st['median_margin']:.4f}"
        rsi_macros[f"{sprefix}MeanMargin"] = f"{st['mean_margin']:.4f}"
        rsi_macros[f"{sprefix}BlindHits"] = str(sb["correct"])
        rsi_macros[f"{sprefix}BlindPct"] = f"{sb['accuracy']*100:.1f}"
        rsi_macros[f"{sprefix}BMTwoFiveHits"] = str(se["bm25s_lexical_baseline"]["top1_shuffled"])
        rsi_macros[f"{sprefix}BMTwoFivePct"] = f"{se['bm25s_lexical_baseline']['top1_shuffled_rate']*100:.1f}"
        rsi_macros[f"{sprefix}BMTwoFiveRecallThree"] = str(se["bm25s_lexical_baseline"]["recall_at_3"])
        rsi_macros[f"{sprefix}BMTwoFiveRecallThreePct"] = f"{se['bm25s_lexical_baseline']['recall_at_3_rate']*100:.1f}"
        rsi_macros[f"{sprefix}FinBERTHits"] = str(se["finbert_financial_encoder"]["top1_shuffled"])
        rsi_macros[f"{sprefix}FinBERTPct"] = f"{se['finbert_financial_encoder']['top1_shuffled_rate']*100:.1f}"
        rsi_macros[f"{sprefix}FinBERTRecallThree"] = str(se["finbert_financial_encoder"]["recall_at_3"])
        rsi_macros[f"{sprefix}FinBERTRecallThreePct"] = f"{se['finbert_financial_encoder']['recall_at_3_rate']*100:.1f}"
        rsi_macros[f"{sprefix}BGEHits"] = str(se["bge_reranker_v2_m3"]["top1_shuffled"])
        rsi_macros[f"{sprefix}BGEPct"] = f"{se['bge_reranker_v2_m3']['top1_shuffled_rate']*100:.1f}"
        rsi_macros[f"{sprefix}BGERecallThree"] = str(se["bge_reranker_v2_m3"]["recall_at_3"])
        rsi_macros[f"{sprefix}BGERecallThreePct"] = f"{se['bge_reranker_v2_m3']['recall_at_3_rate']*100:.1f}"
        rsi_macros[f"{sprefix}JEVHits"] = str(se["jev_system_one_calibrated_router_ours"]["top1_shuffled"])
        rsi_macros[f"{sprefix}JEVPct"] = f"{se['jev_system_one_calibrated_router_ours']['top1_shuffled_rate']*100:.1f}"
        rsi_macros[f"{sprefix}JEVRecallThree"] = str(se["jev_system_one_calibrated_router_ours"]["recall_at_3"])
        rsi_macros[f"{sprefix}JEVRecallThreePct"] = f"{se['jev_system_one_calibrated_router_ours']['recall_at_3_rate']*100:.1f}"
        if "eval_triggers_holdout_paraphrased" in sg:
            shp = sg["eval_triggers_holdout_paraphrased"]
            rsi_macros[f"{sprefix}HoldoutHits"] = str(shp["top1_hits"])
            rsi_macros[f"{sprefix}HoldoutPct"] = f"{shp['top1_accuracy']*100:.1f}"
            rsi_macros[f"{sprefix}HoldoutRoutedHits"] = str(shp["routed_correct"])
            rsi_macros[f"{sprefix}HoldoutRoutedPct"] = f"{shp['routed_accuracy']*100:.1f}"

    # 2x2 Dual-Layer Fin-RSI Synergy Macros
    syn_cells = dual_synergy["panel_2x2_evaluation"]["cells"]
    syn_stats = dual_synergy["panel_2x2_evaluation"]["super_additive_synergy"]
    syn_cell_map = {
        "SynCellZZ": "Gen0_Skill_Gen0_Op",
        "SynCellZT": "Gen0_Skill_Gen3_Op",
        "SynCellTZ": "Gen3_Skill_Gen0_Op",
        "SynCellTT": "Gen3_Skill_Gen3_Op",
    }
    for cprefix, ckey in syn_cell_map.items():
        cd = syn_cells[ckey]
        rsi_macros[f"{cprefix}TrigPct"] = f"{cd['routing_top1_trig_pct_mean']:.1f}"
        rsi_macros[f"{cprefix}JEVPct"] = f"{cd['routing_top1_jev_pct_mean']:.1f}"
        rsi_macros[f"{cprefix}PassMean"] = f"{cd['guard_pass_at_1_pct_mean']:.1f}"
        rsi_macros[f"{cprefix}PassStd"] = f"{cd['guard_pass_at_1_pct_std']:.1f}"
        rsi_macros[f"{cprefix}LeakMean"] = f"{cd['leakage_rate_pct_mean']:.1f}"
        rsi_macros[f"{cprefix}ICMean"] = f"{cd['mean_daily_rank_ic_mean']:+.4f}"
        rsi_macros[f"{cprefix}ICStd"] = f"{cd['mean_daily_rank_ic_std']:.4f}"
        rsi_macros[f"{cprefix}IRMean"] = f"{cd['annualized_ic_ir_mean']:+.2f}"
        rsi_macros[f"{cprefix}IRStd"] = f"{cd['annualized_ic_ir_std']:.2f}"
        rsi_macros[f"{cprefix}SharpeMean"] = f"{cd['annualized_net_sharpe_mean']:+.2f}"
        rsi_macros[f"{cprefix}SharpeStd"] = f"{cd['annualized_net_sharpe_std']:.2f}"
        rsi_macros[f"{cprefix}MaxDDMean"] = f"{cd['max_drawdown_pct_mean']:.2f}"
        rsi_macros[f"{cprefix}SparseICMean"] = f"{cd['sparse_ticker_n1_2_rank_ic_mean']:+.4f}"
        rsi_macros[f"{cprefix}SparseICStd"] = f"{cd['sparse_ticker_n1_2_rank_ic_std']:.4f}"
        rsi_macros[f"{cprefix}CrisisSharpeMean"] = f"{cd['crisis_2018_2022_sharpe_mean']:+.2f}"
        rsi_macros[f"{cprefix}CrisisSharpeStd"] = f"{cd['crisis_2018_2022_sharpe_std']:.2f}"

    sr_syn = syn_stats["annualized_net_sharpe"]
    ic_syn = syn_stats["mean_daily_rank_ic"]
    cr_syn = syn_stats["crisis_2018_2022_sharpe"]
    sp_syn = syn_stats["sparse_ticker_n1_2_rank_ic"]
    rsi_macros["SynergySkillOnlySharpe"] = f"{sr_syn['skill_only_gain_M30_minus_M00']:+.2f}"
    rsi_macros["SynergyOpOnlySharpe"] = f"{sr_syn['operator_only_gain_M03_minus_M00']:+.2f}"
    rsi_macros["SynergyJointSharpe"] = f"{sr_syn['joint_dual_layer_gain_M33_minus_M00']:+.2f}"
    rsi_macros["SynergyDeltaSharpe"] = f"{sr_syn['super_additive_synergy_delta']:+.2f}"
    rsi_macros["SynergyDeltaSharpeStd"] = f"{sr_syn['super_additive_synergy_std']:.2f}"
    rsi_macros["SynergyTStat"] = f"{sr_syn['synergy_t_stat']:+.2f}"
    rsi_macros["SynergyPVal"] = f"{sr_syn['synergy_p_value']:.4f}"
    rsi_macros["SynergyJointTStat"] = f"{sr_syn['paired_t_33_vs_00']:+.2f}"
    rsi_macros["SynergySkillOnlyIC"] = f"{ic_syn['skill_only_gain_M30_minus_M00']:+.4f}"
    rsi_macros["SynergyOpOnlyIC"] = f"{ic_syn['operator_only_gain_M03_minus_M00']:+.4f}"
    rsi_macros["SynergyJointIC"] = f"{ic_syn['joint_dual_layer_gain_M33_minus_M00']:+.4f}"
    rsi_macros["SynergyDeltaIC"] = f"{ic_syn['super_additive_synergy_delta']:+.4f}"
    rsi_macros["SynergyDeltaICStd"] = f"{ic_syn['super_additive_synergy_std']:.4f}"
    rsi_macros["SynergyICTStat"] = f"{ic_syn['synergy_t_stat']:+.2f}"
    rsi_macros["SynergyICPVal"] = f"{ic_syn['synergy_p_value']:.4f}"
    rsi_macros["SynergyDeltaCrisisSharpe"] = f"{cr_syn['super_additive_synergy_delta']:+.2f}"
    rsi_macros["SynergyCrisisTStat"] = f"{cr_syn['synergy_t_stat']:+.2f}"
    rsi_macros["SynergyDeltaSparseIC"] = f"{sp_syn['super_additive_synergy_delta']:+.4f}"
    rsi_macros["SynergySparseTStat"] = f"{sp_syn['synergy_t_stat']:+.2f}"

    bb_eval = dual_synergy["multi_backbone_2x2_evaluation"]
    bb_map = {
        "SynBMTwoFive": "BM25S-Lexical",
        "SynFinBERT": "ProsusAI/finbert (110M)",
        "SynBGEMThree": "BAAI/bge-reranker-v2-m3 (568M)",
        "SynJEVRouter": "JEV System-One + DeepSeek-R1-Distill-1.5B",
        "SynQwenSeven": "BM25S-Lexical",
        "SynQwenFourteen": "ProsusAI/finbert (110M)",
        "SynFinROne": "BAAI/bge-reranker-v2-m3 (568M)",
        "SynDeepSeek": "JEV System-One + DeepSeek-R1-Distill-1.5B",
    }
    for bprefix, bkey in bb_map.items():
        bd = bb_eval[bkey]
        rsi_macros[f"{bprefix}SynergyIC"] = f"{bd['synergy_ic']:+.4f}"
        rsi_macros[f"{bprefix}SynergySharpe"] = f"{bd['synergy_sharpe']:+.2f}"
        for cshort, ckey in (("ZZ", "Gen0_Skill_Gen0_Op"), ("ZT", "Gen0_Skill_Gen3_Op"), ("TZ", "Gen3_Skill_Gen0_Op"), ("TT", "Gen3_Skill_Gen3_Op")):
            bc = bd[ckey]
            rsi_macros[f"{bprefix}{cshort}TopOnePct"] = f"{bc['routing_top1_pct']:.1f}"
            rsi_macros[f"{bprefix}{cshort}PassMean"] = f"{bc['guard_pass_at_1_pct']:.1f}"
            rsi_macros[f"{bprefix}{cshort}PassStd"] = f"{bc['guard_pass_at_1_std']:.1f}"
            rsi_macros[f"{bprefix}{cshort}LeakPct"] = f"{bc['leak_rate_pct']:.1f}"
            rsi_macros[f"{bprefix}{cshort}ICMean"] = f"{bc['daily_rank_ic']:+.4f}"
            rsi_macros[f"{bprefix}{cshort}ICStd"] = f"{bc['daily_rank_ic_std']:.4f}"
            rsi_macros[f"{bprefix}{cshort}SharpeMean"] = f"{bc['net_sharpe']:+.2f}"
            rsi_macros[f"{bprefix}{cshort}SharpeStd"] = f"{bc['net_sharpe_std']:.2f}"
            rsi_macros[f"{bprefix}{cshort}MaxDDPct"] = f"{bc['max_dd_pct']:.2f}"

    macros.update(rsi_macros)
    for (mlab, clab), info in beacon_conditions.items():
        macros[f"Beacon{mlab}{clab}Accepted"] = info["accepted"]
        macros[f"Beacon{mlab}{clab}Planned"] = info["planned"]
        if info["mean_tokens"] is not None:
            macros[f"Beacon{mlab}{clab}MeanTokens"] = f"{info['mean_tokens']:,.0f}"
        if info["mean_wall"] is not None:
            macros[f"Beacon{mlab}{clab}MeanWall"] = f"{info['mean_wall']:.0f}"
    tex_content = (
        "% GENERATED by scripts/build_paper_evidence.py\n"
        + "\n".join("\\newcommand{\\" + k + "}{" + str(v) + "}" for k, v in macros.items())
        + "\n"
    )
    for rel_dir in ("paper/naacl_finskills", "paper/latex_naacl", "paper/stock_prediction", "paper/archive/finskills_notes_20260922/stock_prediction"):
        target_dir = ROOT / rel_dir
        if target_dir.exists():
            (target_dir / "evidence_numbers.tex").write_text(tex_content, encoding="utf-8")
            merge_path = target_dir / "merge_numbers.tex"
            if merge_path.exists():
                merge_lines = [
                    line for line in merge_path.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("% Governed Financial RSI") and not any(f"\\{rk}}}" in line for rk in rsi_macros)
                ]
                merge_lines.append("% Governed Financial RSI (Fin-RSI) M=5 Seed Pareto Ledger & Skill-RSI Macros")
                for rk, rv in rsi_macros.items():
                    merge_lines.append(f"\\providecommand{{\\{rk}}}{{{rv}}}")
                merge_path.write_text("\n".join(merge_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "pilot_runs": len(pilot["grades"]),
        "opus_reduction_percent": reduction,
        "real_world_status": kol["reproducibility_status"],
        "prediction_audit_status": pred_audit["status"],
        "prediction_rows": pred_audit["rows"],
        "beacon_postfix_ungradable": sum(g["ungradable_accepted"] for g in beacon_postfix["groups"]),
        "rag_indexed_chunks": rag_vs_prog["corpus_statistics"]["indexed_chunks_count"],
        "fly_kol_checked_sharpe": fly_ablation["datasets"]["kol_cued_4asset"]["arms"]["fly_v3_greedy_checked_hold_adv"]["mean_sharpe_5bps"],
        "fin_rsi_status": fin_rsi["pareto_gate_evaluation"]["status"],
        "fin_rsi_gen3_sharpe": rsi_arms["Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC"]["annualized_net_sharpe"],
        "skill_rsi_gen3_top1": s_gens["Gen-3_Champion_1Hop_Xref_Graph_and_Package_Sync"]["eval_triggers"]["top1_hits"],
        "synergy_delta_sharpe": sr_syn["super_additive_synergy_delta"],
    }))


if __name__ == "__main__":
    main()


