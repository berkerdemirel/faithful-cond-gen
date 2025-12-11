from faithful_cond_gen.data.rxrx1 import RxRx1Config, RxRx1Dataset


def main():
    cfg = RxRx1Config(root="/path/to/rxrx1")  # placeholder, not used yet
    ds = RxRx1Dataset(cfg)
    print("RxRx1 config:", cfg)
    print("Dataset length:", len(ds))


if __name__ == "__main__":
    main()
