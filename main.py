
if __name__ == '__main__'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str)
    args = parser.parse_args()
    data_set = args.data_set
    begin_train(load_model=False, need_to_save_the_process=True, data_set=data_set)
