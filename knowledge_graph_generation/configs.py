#!/usr/bin/python3

huggingface_token = '

tgi_host = ""

neo4j_host = ""
neo4j_user = ""
neo4j_password = ""
neo4j_db = ""

use_fewshot = False
use_selector = False

node_types = ['Patent', 'Inventor', 'Applicant', 'Agent', 'Technical Field', 'Patent Owner', 'Patent Citation']

rel_types = [
  ('Patent', 'Invented_by', 'Inventor'),
  ('Patent', 'Applied_by', 'Applicant'),
  ('Patent', 'Processed_by', 'Agent'),
  ('Patent', 'Belongs_to', 'Technical Field'),
  ('Patent', 'Owned_by', 'Patent Owner'),
  ('Patent', 'Cite', 'Patent')
]

examples = [
]
