program reference_single_tree_driver
  use sklearn_common,only:dp,require
  use sklearn_hierarchy,only:linkage_tree,hierarchy_result
  use sklearn_sparse,only:csr_from_dense,csr_from_arrays,csr_matrix
  implicit none
  real(dp),allocatable::x(:,:),g(:,:),copy(:,:),gcopy(:,:)
  type(hierarchy_result)::tree
  type(csr_matrix)::csr
  integer,allocatable::ptr(:),cols(:)
  real(dp),allocatable::values(:)
  integer::n,p,mode,clusters,distances,i,ios,j,pos,method,metric
  character(16)::methods(1)=[character(16)::"single"]
  character(16)::metrics(12)=[character(16)::"euclidean","manhattan","chebyshev","cosine", &
    "sqeuclidean","precomputed","l1","l2","cityblock","minkowski","p","infinity"]
  do
    read(*,*,iostat=ios)n,p,mode,clusters,distances,method,metric
    if(ios/=0)exit
    allocate(x(n,p),g(n,n));g=0
    do i=1,n
      read(*,*)x(i,:)
    end do
    if(mode/=0)then
      do i=1,n
        read(*,*)g(i,:)
      end do
    end if
    copy=x;gcopy=g
    if(mode>=3)then
      if(mode==3.or.mode==5)then
        csr=csr_from_dense(g,store_zeros=mode==5)
      else
        allocate(ptr(n+1),cols(2*n*n),values(2*n*n));pos=1;ptr(1)=1
        do i=1,n
          do j=n,1,-1
            if(g(i,j)==0)cycle
            cols(pos:pos+1)=j;values(pos)=.4_dp*g(i,j);values(pos+1)=.6_dp*g(i,j);pos=pos+2
          end do
          ptr(i+1)=pos
        end do
        csr=csr_from_arrays(n,n,ptr,cols(:pos-1),values(:pos-1));deallocate(ptr,cols,values)
      end if
    end if
    select case(mode)
    case(0)
      tree=linkage_tree(x,n_clusters=clusters,return_distance=distances/=0,linkage=methods(method),affinity=metrics(metric))
    case(1)
      tree=linkage_tree(x,g,n_clusters=clusters,return_distance=distances/=0,linkage=methods(method),affinity=metrics(metric))
    case(2)
      tree=linkage_tree(x,g/=0,n_clusters=clusters,return_distance=distances/=0,linkage=methods(method),affinity=metrics(metric))
    case(3:5)
      tree=linkage_tree(x,csr,n_clusters=clusters,return_distance=distances/=0,linkage=methods(method),affinity=metrics(metric))
    end select
    call require(all(x==copy).and.all(g==gcopy),'Ward changed input')
    write(*,'(3(i0,1x),l1,1x,l1)')tree%n_leaves,tree%n_connected_components,size(tree%children,1), &
      tree%structured,tree%connectivity_completed
    do i=1,size(tree%children,1)
      write(*,'(2(i0,1x))')tree%children(i,:)
    end do
    write(*,'(i0)')size(tree%parents)
    write(*,'(*(i0,1x))')tree%parents
    write(*,'(i0)')size(tree%distances)
    write(*,'(*(es26.17e3,1x))')tree%distances
    deallocate(x,g)
  end do
end program
