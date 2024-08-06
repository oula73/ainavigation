for maze in alternating_gaps bugtrap_forest forest gaps_and_forest mazes multiple_bugtraps shifting_gaps single_bugtrap
do
python generate_threat_spp_instances.py --input-path planning-datasets/data/mpd/original/$maze --output-path data/mpd_with_threaten/ --maze-size 32 --mechanism moore
done