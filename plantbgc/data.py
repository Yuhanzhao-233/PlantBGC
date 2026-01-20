from plantbgc import util

# !!!
# !!!
#
# Don't forget to bump up the DATA_RELEASE_VERSION to plantbgc.__version__ when updating the downloads list
# A data release should be published with a regular code release, with the same release version number
# Code releases can happen more often, so the data release version can lag behind the code version
#
DATA_RELEASE_VERSION = '0.1'
#
# !!!
# !!!

PFAM_DB_VERSION = '1.0'
PFAM_DB_FILE_NAME = 'Pfam_{}.hmm'.format(PFAM_DB_VERSION)
PFAM_CLANS_FILE_NAME = 'Pfam.clans_{}.tsv'.format(PFAM_DB_VERSION)

DOWNLOADS = [
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v{}/plantbgc.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'plantbgc.pkl',
        'dir': 'detector',
        'checksum': '87bb056f9cd7b314c4f0e23d668c67fb',
        'versioned': True
    },
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v{}/clusterfinder.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'clusterfinder.pkl',
        'dir': 'detector',
        'checksum': 'bcd328c771f9f0a2ea829409d86ca89e',
        'versioned': True
    },
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v{}/BiLSTM.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'BiLSTM.pkl',
        'dir': 'detector',
        'checksum': '7e9218be79ba45bc9adb23bed3845dc1',
        'versioned': True
    },
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v{}/Transformer_stage1.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'Transformer_stage1.pkl',
        'dir': 'detector',
        'checksum': 'fa9f0a62c84329bca41a711a7d2b3dcb',
        'versioned': True
    },
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v{}/activity_classifier.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'activity_classifier.pkl',
        'dir': 'classifier',
        'checksum': '395ff87ee1e2c2dce0e2ca77de58e408',
        'versioned': True
    },
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v{}/class_classifier.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'class_classifier.pkl',
        'dir': 'classifier',
        'checksum': 'cfbd5b2bc0dd01a8e9e777bdad7c7519',
        'versioned': True
    },
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v0.1/Pfam_{}.hmm'.format(PFAM_DB_VERSION),    #79a3328e4c95b13949a4489b19959fc5
        'target': PFAM_DB_FILE_NAME,
        'gzip': False,
        'after': util.run_hmmpress,
        'checksum': '8340ad7e81c84bb92ce8773eb907e388',
        'versioned': False
    },
    {
        'url': 'https://github.com/Yuhanzhao-233/PlantBGC/releases/download/v0.1/Pfam.clans_{}.tsv'.format(PFAM_DB_VERSION),
        'target': PFAM_CLANS_FILE_NAME,
        'gzip': False,
        'checksum': 'a0a4590ffb2b33b83ef2b28f6ead886b',
        'versioned': False
    }
]
