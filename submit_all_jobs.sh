
./idre_submit_job.sh galkin om single_slit True powerlaw
./idre_submit_job.sh galkin constant single_slit True powerlaw
./idre_submit_job.sh galkin om ifu True powerlaw
./idre_submit_job.sh galkin constant ifu True powerlaw
./idre_submit_job.sh jampy constant ifu True powerlaw
./idre_submit_job.sh jampy om ifu True powerlaw

./idre_submit_job.sh jampy constant ifu False powerlaw
./idre_submit_job.sh jampy om ifu False powerlaw
./idre_submit_job.sh jampy step ifu False powerlaw
./idre_submit_job.sh jampy free_step ifu False powerlaw