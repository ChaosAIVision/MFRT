# BEAD Planning Summary - 2-Phase Meta-Prompt Optimization System

## ✅ Planning Complete

Đã hoàn thành việc phân tích và lập kế hoạch cho hệ thống 2-phase meta-prompt optimization theo yêu cầu trong AGENTS.md.

## 📦 Components Planned (6 components)

### Foundation Layer
1. **XMLOutputParser** (chaos-auto-prompt-4n7) - P0
   - Parse `<construction>` and `<think>` tags from LLM output
   - No dependencies - **READY TO BUILD**
   - Effort: 2-4 hours

### Extraction & Evaluation Layer
2. **ConstructionExtractor** (chaos-auto-prompt-tts) - P1
   - Extract entities, state variables, actions, constraints
   - Depends on: XMLOutputParser
   - Effort: 6-8 hours

3. **ReasoningPathEvaluator** (chaos-auto-prompt-qja) - P1
   - Evaluate reasoning logic paths (good/bad classification)
   - Depends on: XMLOutputParser
   - **READY TO BUILD** (parallel with ConstructionSimilarityEvaluator)
   - Effort: 10-14 hours

### Quality Assessment Layer
4. **ConstructionSimilarityEvaluator** (chaos-auto-prompt-ro2) - P1
   - Measure 90% similarity threshold for Phase 1
   - Depends on: ConstructionExtractor
   - Effort: 8-12 hours

### Meta-Prompt Layer
5. **TwoPhaseMetaPrompt** (chaos-auto-prompt-zo9) - P1
   - Build optimized prompts with good/bad examples
   - Depends on: ConstructionSimilarityEvaluator, ReasoningPathEvaluator
   - Effort: 6-8 hours

### Orchestration Layer
6. **TwoPhaseOptimizer** (chaos-auto-prompt-86c) - P2
   - Main training pipeline coordinator
   - Depends on: All above components
   - Effort: 12-16 hours

**Total Estimated Effort:** 44-62 hours

## 📋 Build Order (Optimal Sequence)

```
Phase 1: XMLOutputParser (4n7)
         ↓
Phase 2: ConstructionExtractor (tts) + ReasoningPathEvaluator (qja) [PARALLEL]
         ↓
Phase 3: ConstructionSimilarityEvaluator (ro2)
         ↓
Phase 4: TwoPhaseMetaPrompt (zo9)
         ↓
Phase 5: TwoPhaseOptimizer (86c)
```

## 🎯 Ready to Implement Now

Run this command to see components ready for implementation:
```bash
bd ready
```

Currently ready (no blockers):
1. **XMLOutputParser** (P0) - Foundation component
2. **ReasoningPathEvaluator** (P1) - Can build in parallel

## 📁 Documentation Created

All plan documents in `report_component_task/`:
- ✅ `task_00_overview.md` - Project overview with dependency graph
- ✅ `task_xml_output_parser.md` - XML tag parsing component
- ✅ `task_construction_extractor.md` - Construction element extraction
- ✅ `task_construction_similarity_evaluator.md` - Phase 1 evaluation
- ✅ `task_reasoning_path_evaluator.md` - Phase 2 evaluation
- ✅ `task_two_phase_meta_prompt.md` - Meta-prompt builder
- ✅ `task_two_phase_optimizer.md` - Training orchestrator

## 🔑 Key Design Decisions

### Phase 1: Construction Optimization
- Target: 90% similarity with groundtruth construction
- Method: Iterative meta-prompt refinement with feedback
- Evaluator: ConstructionSimilarityEvaluator (hybrid: Jaccard + fuzzy + semantic)
- Max iterations: 10

### Phase 2: Reasoning Optimization
- Target: Maximize answer correctness
- Method: Good/bad reasoning path tracking
- Evaluator: ReasoningPathEvaluator (logic classification)
- Max iterations: 20

### Technical Stack
- **Semantic Similarity**: sentence-transformers (all-MiniLM-L6-v2)
- **Fuzzy Matching**: rapidfuzz
- **Templates**: Jinja2 or f-strings
- **Persistence**: JSON files
- **Providers**: Existing OpenAI/Google providers

## 🎓 BEAD Methodology Applied

### Break down (Phân tách)
✅ Broke down 2-phase system into 6 modular components
✅ Each component has single, clear responsibility
✅ Clean boundaries and interfaces defined

### Explain (Giải thích)
✅ Each component has detailed "What/Why/Where It Fits" sections
✅ Data flow and relationships documented
✅ Purpose in bigger picture explained

### Analyze (Phân tích)
✅ Input/output specifications with exact field types
✅ Dependencies mapped and visualized
✅ Edge cases and risks identified for each component

### Document (Tài liệu hóa)
✅ 7 comprehensive plan documents created
✅ Testing strategy defined for each component
✅ Missing information flagged for follow-up

## ⚠️ Critical Missing Information

Before implementation, need to gather:

1. **Real Dataset** - Training examples with:
   - Problem statements
   - Groundtruth constructions
   - Groundtruth reasoning paths
   - Groundtruth answers

2. **Performance Baselines** - Current system metrics

3. **Budget Constraints** - Cost limits per run

4. **Language Requirements** - Vietnamese support needed?

## 🚀 Next Steps

### Option 1: Start Implementation (Mode 2)
```bash
# Mark XMLOutputParser as in-progress
bd update chaos-auto-prompt-4n7 --status in_progress

# Read the plan
cat report_component_task/task_xml_output_parser.md

# Implement, test, close
```

### Option 2: Review Plans First
- Review all plan documents
- Ask questions about unclear requirements
- Gather missing information
- Validate assumptions

### Option 3: Full Cycle (Mode 3)
- Review and approve plans
- Implement all components in build order
- Test each component with real data
- Create implementation reports

## 📊 Dependency Verification

Run this to see full dependency tree:
```bash
bd dep tree chaos-auto-prompt-86c
```

Expected output shows clean dependency chain from XMLOutputParser → TwoPhaseOptimizer.

## 🔍 Quality Checklist

- [x] All 6 components have detailed plans
- [x] Dependencies correctly mapped in bd
- [x] Build order respects dependencies
- [x] Input/output specs are detailed
- [x] Test cases planned for each component
- [x] Edge cases and risks identified
- [x] Technology choices justified
- [x] Effort estimates provided
- [x] Overview document with architecture
- [x] Mermaid dependency graph created

## 💡 Recommendations

1. **Start with XMLOutputParser** - Foundation component, simple, high value
2. **Test with real LLM outputs early** - Validate parsing assumptions
3. **Implement incrementally** - Build → Test → Report → Next component
4. **Gather dataset ASAP** - Critical for realistic testing
5. **Use sentence-transformers carefully** - Monitor performance/memory
6. **Version control path databases** - Track changes in good/bad paths

---

**Status:** ✅ Planning Complete - Ready for Implementation

**Created:** 2026-01-28

**Method:** BEAD (Break down, Explain, Analyze, Document) + bd tool integration
