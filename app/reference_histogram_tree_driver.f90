program reference_histogram_tree_driver
  use sklearn_common, only: dp
  use sklearn_histogram_binning
  use sklearn_histogram_tree
  use, intrinsic :: iso_fortran_env, only: int16,int64,real32
  implicit none
  type(histogram_bin_mapper) :: mapper
  type(histogram_tree) :: tree
  integer :: n,p,m,min_leaf,max_leaves,max_depth,j
  integer(int64) :: seed
  integer, allocatable :: monotonic(:)
  integer(int16), allocatable :: bins(:,:)
  real(dp) :: l2,rate
  real(dp), allocatable :: x(:,:),q(:,:),prediction(:)
  real(real32), allocatable :: g(:),h(:)
  read(*,*) n,p,m,mapper%max_bins,min_leaf,max_leaves,max_depth,l2,rate,seed
  allocate(x(n,p),q(m,p),g(n),h(n),monotonic(p))
  do j=1,n
    read(*,*) x(j,:)
  end do
  do j=1,m
    read(*,*) q(j,:)
  end do
  read(*,*) g;read(*,*) h;read(*,*) monotonic
  call mapper%fit(x,seed);bins=mapper%transform(x)
  tree=grow_histogram_tree(bins,g,h,mapper,min_leaf,max_leaves,max_depth,l2,rate,monotonic)
  print *,size(tree%nodes_)
  print *,mapper%n_bins_non_missing_
  print *,bins
  do j=1,p
    print '( *(es25.16e3,1x))',mapper%bin_thresholds_(j)%values
  end do
  prediction=tree%predict(q);print '( *(es25.16e3,1x))',prediction
  do j=1,size(tree%nodes_)
    associate(node=>tree%nodes_(j))
      print *,node%feature,node%bin_threshold,node%left,node%right,node%count,node%depth, &
        merge(1,0,node%missing_go_to_left)
      print '( *(es25.16e3,1x))',node%value,node%gain,node%threshold
    end associate
  end do
end program
