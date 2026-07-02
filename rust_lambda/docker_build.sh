docker run --rm -v "${PWD%/*}":/work -w /work public.ecr.aws/sam/build-provided.al2023 sh -c "sh rust_lambda/docker_cmds.sh"

cp ./target/release/rust_lambda bootstrap
zip lambda.zip bootstrap