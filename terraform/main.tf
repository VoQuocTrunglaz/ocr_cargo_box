# Tạo VPC
resource "aws_vpc" "my_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "MyVPC"
  }
}

# Public Subnet
resource "aws_subnet" "public_subnet" {
  vpc_id                  = aws_vpc.my_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "ap-southeast-1a"

  tags = {
    Name = "PublicSubnet"
  }
}

# Private Subnet
resource "aws_subnet" "private_subnet" {
  vpc_id     = aws_vpc.my_vpc.id
  cidr_block = "10.0.2.0/24"
  availability_zone = "ap-southeast-1a"
  
  tags = {
    Name = "PrivateSubnet"
  }
}

# Internet Gateway for Public Subnet
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.my_vpc.id

  tags = {
    Name = "InternetGateway"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.my_vpc.id

  tags = {
    Name = "PublicRouteTable"
  }
}

# Route cho Public Subnet đi ra Internet
resource "aws_route" "public_internet_access" {
  route_table_id         = aws_route_table.public_rt.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.igw.id
}

# Gán Route Table cho Public Subnet
resource "aws_route_table_association" "public_assoc" {
  subnet_id      = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_rt.id
}

# Route Table cho Private Subnet (Không có Internet Gateway)
resource "aws_route_table" "private_rt" {
  vpc_id = aws_vpc.my_vpc.id

  tags = {
    Name = "PrivateRouteTable"
  }
}

# Gán Route Table cho Private Subnet
resource "aws_route_table_association" "private_assoc" {
  subnet_id      = aws_subnet.private_subnet.id
  route_table_id = aws_route_table.private_rt.id
}

# NAT Instance trong Public Subnet
resource "aws_instance" "nat_instance" {
  ami                         = "ami-0a72af05d27b49ccb" # AMI Amazon Linux 2 (ap-southeast-1)
  instance_type               = "t2.micro"
  subnet_id                   = aws_subnet.public_subnet.id
  associate_public_ip_address = true
  key_name                    = "myapp"
  source_dest_check           = false # Tắt source/destination check để NAT hoạt động
  #security_groups             = [aws_security_group.nat_sg.id]

  user_data = <<-EOF
              #!/bin/bash
              echo 1 > /proc/sys/net/ipv4/ip_forward
              iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
              yum install -y iptables-services
              service iptables save
              EOF

  tags = {
    Name = "NATInstance"
  }
}

# Route cho Private Subnet qua NAT Instance
resource "aws_route" "private_nat_route" {
  route_table_id         = aws_route_table.private_rt.id
  destination_cidr_block = "0.0.0.0/0"
  instance_id            = aws_instance.nat_instance.id
}

# Gắn Security Group cho NAT Instance
resource "aws_network_interface_sg_attachment" "nat_sg_attachment" {
  security_group_id    = aws_security_group.nat_sg.id
  network_interface_id = aws_instance.nat_instance.primary_network_interface_id
}

# Security Group cho NAT Instance
resource "aws_security_group" "nat_sg" {
  vpc_id = aws_vpc.my_vpc.id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # Cho phép mọi traffic từ Private Subnet
    cidr_blocks = ["10.0.2.0/24"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "NATSecurityGroup"
  }
}

# Security Group cho EC2 Public
resource "aws_security_group" "public_sg" {
  vpc_id = aws_vpc.my_vpc.id

# Cho phép SSH từ mọi nơi
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] 
  }

# Cho phép HTTP (cổng 80) từ mọi nơi
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

# Cho phép HTTPS (cổng 443) từ mọi nơi
  ingress {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }

  # # Cho phép app (cổng 3000) từ mọi nơi
  # ingress {
  #     from_port   = 3000
  #     to_port     = 3000
  #     protocol    = "tcp"
  #     cidr_blocks = ["0.0.0.0/0"]
  #   }

  ingress {
      from_port   = 8501
      to_port     = 8501
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "PublicEC2SecurityGroup"
  }
}

# Security Group cho EC2 Private
resource "aws_security_group" "private_sg" {
  vpc_id = aws_vpc.my_vpc.id

# Chỉ cho phép SSH từ EC2 Public
  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.public_sg.id]
  }

  ingress {
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.public_sg.id]
  }
  
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.public_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "PrivateEC2SecurityGroup"
  }
}

# IAM Role cho EC2 Instances
resource "aws_iam_role" "ec2_role" {
  name = "aws_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Gắn IAM Role cho EC2 
resource "aws_iam_role_policy_attachment" "ec2_role_policy" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "ec2_instance_profile"
  role = aws_iam_role.ec2_role.name
}

# EC2 Public Instance
resource "aws_instance" "public_ec2" {
  ami                    = "ami-0a72af05d27b49ccb"
  instance_type          = "t2.micro"
  key_name               = "myapp"
  subnet_id              = aws_subnet.public_subnet.id
  security_groups        = [aws_security_group.public_sg.id]
  associate_public_ip_address = true
  iam_instance_profile    = aws_iam_instance_profile.ec2_profile.name

  tags = {
    Name = "PublicEC2Instance"
  }
}

# EC2 Private Instance
resource "aws_instance" "private_ec2" {
  ami           = "ami-0a72af05d27b49ccb"
  instance_type = "t2.micro"
  key_name      = "myapp"
  subnet_id     = aws_subnet.private_subnet.id
  security_groups = [aws_security_group.private_sg.id]
  private_ip    = "10.0.2.100"
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  root_block_device {
    volume_size = 16           
    volume_type = "gp3"       
  }

  tags = {
    Name = "PrivateEC2Instance"
  }
}
# Private Endpoint (S3)
resource "aws_vpc_endpoint" "s3_endpoint" {
  vpc_id            = aws_vpc.my_vpc.id
  service_name      = "com.amazonaws.ap-southeast-1.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.public_rt.id, aws_route_table.private_rt.id]

  tags = {
    Name = "S3PrivateEndpoint"
  }
}