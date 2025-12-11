#!/usr/bin/env python3
from goalden import GoaldenOSv4_2

# Quick 10-second test on a synthetic black hole (outward star)
detector = GoaldenOSv4_2()
detector.generate_synthetic("outward_star", n=500)
detector.analyze(mode="path_robust")
detector.visualize()          # shows the scary red gauge