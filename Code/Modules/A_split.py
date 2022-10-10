from glob import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
from Code.Loaders.ImageLoader import ImageLoader


def split(args, feature):

    """Split dataset into train, test and validation.

    Variables and default values:

    test_size=0.2
    validation_size=0.2
    seed=42

    """

    # Retrieve images
    images = glob("Input/Images/*.jpg")

    # Retrieve labels
    labels = [
        image.replace("Images", "Labels").replace("jpg", "yaml") for image in images
    ]

    # Separate test and train set
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=args.test_size, random_state=args.seed
    )

    # Separate train and validation set
    images_train, images_validation, labels_train, labels_validation = train_test_split(
        images_train,
        labels_train,
        test_size=args.validation_size,
        random_state=args.seed,
    )

    # Initialise datasets
    train_set = ImageLoader(
        images_train,
        labels_train,
        set_type="train",
        batch_size=args.batch,
        feature=feature,
    )
    validation_set = ImageLoader(
        images_validation,
        labels_validation,
        set_type="validation",
        batch_size=args.batch,
        maximum=train_set.maximum,
        minimum=train_set.minimum,
        feature=feature,
    )
    test_set = ImageLoader(
        images_test,
        labels_test,
        set_type="test",
        batch_size=args.batch,
        maximum=train_set.maximum,
        minimum=train_set.minimum,
        feature=feature,
    )

    # Save splitting in .yaml files
    train_set.save_split()
    validation_set.save_split()
    test_set.save_split()

    return train_set, validation_set, test_set
