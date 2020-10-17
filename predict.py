from address_parser.model import predict_one

model_filename = 'model1.ckpt'
predict_one("1600 Amphitheatre Pkwy, Mountain View, CA 94043, United States", model_filename)