export teacher_dir="/mnt/data-zeus2/mtreviso/meta-expl-experiments/TrainTeacher/Baseline.baseline/teacher_dir"
export student_dir="/mnt/data-zeus2/mtreviso/meta-expl-experiments/TrainStudent/KLDCoeff.1+MetaInterval.2+MetaLR.1e2+MetaLearn.true+NumExamples.2000+Pivot.true+Seed.1+StudentExplainer.softmax_param+Teache
rExplainer.softmax_param/student_dir"
export teacher_expl_dir="/mnt/data-zeus2/mtreviso/meta-expl-experiments/TrainStudent/KLDCoeff.1+MetaInterval.2+MetaLR.1e2+MetaLearn.true+NumExamples.2000+Pivot.true+Seed.1+StudentExplainer.softmax_param+T
eacherExplainer.softmax_param/teacher_expl_dir"
export student_expl_dir="/mnt/data-zeus2/mtreviso/meta-expl-experiments/TrainStudent/KLDCoeff.1+MetaInterval.2+MetaLR.1e2+MetaLearn.true+NumExamples.2000+Pivot.true+Seed.1+StudentExplainer.softmax_param+T
eacherExplainer.softmax_param/student_expl_dir"
export results="/mnt/data-zeus2/mtreviso/meta-expl-experiments/TrainStudent/KLDCoeff.1+MetaInterval.2+MetaLR.1e2+MetaLearn.true+NumExamples.2000+Pivot.true+Seed.1+StudentExplainer.softmax_param+TeacherExp
lainer.softmax_param/results"
export submitter="cuda_async"
export meta_interval="2"
export kld_coeff="1."
export mem="16000"
export gpus="1"
export cpus="2"
export meta_lr="1e-2"
export teacher_explainer="softmax_param"
export repo="/home/mtreviso/meta-expl"
export student_explainer="softmax_param"
export metalearn="true"
export seed="1"
export pivot="true"
export num_examples="2000"

python $repo/meta_expl/train.py \
  --model-type student \
  --teacher-explainer "$teacher_explainer" \
  --student-explainer "$student_explainer" \
  --num-examples "$num_examples" \
  --kld-coeff "$kld_coeff" \
  --metalearn-interval "$meta_interval" \
  --meta-lr "$meta_lr" \
  --pivot \
  --teacher-dir "$teacher_dir" \
  --model-dir "$student_dir" \
  --teacher-explainer-dir "$teacher_expl_dir" \
  --explainer-dir "$student_expl_dir" \
  --seed $seed
