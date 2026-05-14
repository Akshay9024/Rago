from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class HardwareProfile:
    gpu_type: Literal["rtx4090", "a6000_48g", "a100_40g", "a100_80g", "h100_80g"]
    vram_gb: int
    precision: Literal["int8", "bfloat16", "float16", "fp8"]
    quantization: Optional[Literal["bitsandbytes", "awq", "gptq"]]
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_batch_size: int = 32
    decode_batch_size: int = 64
    retrieval_batch_size: int = 4


RTX4090 = HardwareProfile(
    gpu_type="rtx4090",
    vram_gb=24,
    precision="float16",
    quantization="bitsandbytes",
    tensor_parallel_size=1,
    max_batch_size=16,
    decode_batch_size=32,
    retrieval_batch_size=4,
)

A6000_48G = HardwareProfile(
    gpu_type="a6000_48g",
    vram_gb=48,
    precision="bfloat16",
    quantization=None,
    tensor_parallel_size=1,
    max_batch_size=48,
    decode_batch_size=96,
    retrieval_batch_size=12,
)

A100_40G = HardwareProfile(
    gpu_type="a100_40g",
    vram_gb=40,
    precision="bfloat16",
    quantization=None,
    tensor_parallel_size=1,
    max_batch_size=32,
    decode_batch_size=64,
    retrieval_batch_size=8,
)

A100_80G = HardwareProfile(
    gpu_type="a100_80g",
    vram_gb=80,
    precision="bfloat16",
    quantization=None,
    tensor_parallel_size=1,
    max_batch_size=64,
    decode_batch_size=128,
    retrieval_batch_size=16,
)

H100_80G = HardwareProfile(
    gpu_type="h100_80g",
    vram_gb=80,
    precision="bfloat16",
    quantization=None,
    tensor_parallel_size=2,
    max_batch_size=128,
    decode_batch_size=256,
    retrieval_batch_size=32,
)


@dataclass
class DeploymentMode:
    mode: Literal["single_node", "multi_node", "io_net"]
    local_rank: int = 0
    world_size: int = 1


@dataclass
class InfraConfig:
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_cold_collection: str = "cold_corpus"
    qdrant_api_key: Optional[str] = None

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_max_memory_gb: int = 20

    nats_url: str = "local"
    nats_stream: str = "iterative_rag"
    nats_consumer_name: str = "decode_worker"
    nats_dedup_window_seconds: int = 30

    grpc_host: str = "localhost"
    grpc_port: int = 50051

    ray_address: Optional[str] = None
    haproxy_bind: str = "0.0.0.0:80"


@dataclass
class SystemConfig:
    hardware: HardwareProfile
    deployment: DeploymentMode
    infra: InfraConfig
    model_dir: str = "/Volumes/Ak_T7/models"
    log_level: str = "INFO"

    @classmethod
    def for_rtx4090_local(cls) -> "SystemConfig":
        return cls(
            hardware=RTX4090,
            deployment=DeploymentMode(mode="single_node"),
            infra=InfraConfig(),
        )

    @classmethod
    def for_a6000_local(cls) -> "SystemConfig":
        return cls(
            hardware=A6000_48G,
            deployment=DeploymentMode(mode="single_node"),
            infra=InfraConfig(),
        )

    @classmethod
    def for_a6000_ssh(cls, host: str) -> "SystemConfig":
        return cls(
            hardware=A6000_48G,
            deployment=DeploymentMode(mode="single_node"),
            infra=InfraConfig(
                qdrant_host=host,
                redis_host=host,
                nats_url=f"nats://{host}:4222",
                grpc_host=host,
            ),
        )

    @classmethod
    def for_a100_ssh(cls, host: str) -> "SystemConfig":
        return cls(
            hardware=A100_40G,
            deployment=DeploymentMode(mode="single_node"),
            infra=InfraConfig(
                qdrant_host=host,
                redis_host=host,
                nats_url=f"nats://{host}:4222",
                grpc_host=host,
            ),
        )

    @classmethod
    def for_a100_80g_ssh(cls, host: str) -> "SystemConfig":
        return cls(
            hardware=A100_80G,
            deployment=DeploymentMode(mode="single_node"),
            infra=InfraConfig(
                qdrant_host=host,
                redis_host=host,
                nats_url=f"nats://{host}:4222",
                grpc_host=host,
            ),
        )

    @classmethod
    def for_h100_ssh(cls, host: str) -> "SystemConfig":
        return cls(
            hardware=H100_80G,
            deployment=DeploymentMode(mode="single_node"),
            infra=InfraConfig(
                qdrant_host=host,
                redis_host=host,
                nats_url=f"nats://{host}:4222",
                grpc_host=host,
            ),
        )

    @classmethod
    def for_io_net_cluster(cls, coordinator_host: str, world_size: int = 4) -> "SystemConfig":
        return cls(
            hardware=H100_80G,
            deployment=DeploymentMode(mode="io_net", world_size=world_size),
            infra=InfraConfig(
                qdrant_host=coordinator_host,
                redis_host=coordinator_host,
                nats_url=f"nats://{coordinator_host}:4222",
                grpc_host=coordinator_host,
                ray_address=f"ray://{coordinator_host}:10001",
            ),
        )
