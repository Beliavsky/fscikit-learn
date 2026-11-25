program reference_linkage_fit_driver
  use sklearn_common,only:dp,require
  use sklearn_cluster,only:agglomerative_clustering,feature_agglomeration
  use sklearn_sparse,only:csr_from_dense,csr_from_arrays
  implicit none
  type(agglomerative_clustering)::model
  type(feature_agglomeration)::features
  real(dp),allocatable::x(:,:),g(:,:),copy(:,:),z(:,:),back(:,:),values(:)
  integer,allocatable::ptr(:),cols(:)
  integer::n,p,mode,clusters,full,distances,feature,pooling,i,j,pos,ios,method,metric
  character(16)::methods(2)=[character(16)::"complete","average"]
  character(16)::metrics(10)=[character(16)::"euclidean","manhattan","chebyshev","cosine", &
    "sqeuclidean","precomputed","l1","l2","cityblock","minkowski"]
  real(dp)::threshold
  do
    read(*,*,iostat=ios)n,p,mode,clusters,full,distances,threshold,feature,pooling,method,metric
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
    copy=x
    ! Reuse each object across cases, clearing optional configuration explicitly.
    if(allocated(model%connectivity))deallocate(model%connectivity)
    if(allocated(model%connectivity_mask))deallocate(model%connectivity_mask)
    if(allocated(model%connectivity_csr))deallocate(model%connectivity_csr)
    if(allocated(model%distance_threshold))deallocate(model%distance_threshold)
    model%n_clusters=clusters;model%compute_distances=distances/=0
    model%linkage=methods(method);model%metric=metrics(metric)
    select case(full)
    case(0);model%compute_full_tree='auto'
    case(1);model%compute_full_tree='true'
    case(2);model%compute_full_tree='false'
    end select
    if(threshold>=0)model%distance_threshold=threshold
    select case(mode)
    case(1);model%connectivity=g
    case(2);model%connectivity_mask=g/=0
    case(3);model%connectivity_csr=csr_from_dense(g)
    case(4)
      allocate(ptr(n+1),cols(2*n*n),values(2*n*n));pos=1;ptr(1)=1
      do i=1,n
        do j=n,1,-1
          if(g(i,j)==0)cycle
          cols(pos:pos+1)=j;values(pos)=.4_dp*g(i,j);values(pos+1)=.6_dp*g(i,j);pos=pos+2
        end do
        ptr(i+1)=pos
      end do
      model%connectivity_csr=csr_from_arrays(n,n,ptr,cols(:pos-1),values(:pos-1))
      deallocate(ptr,cols,values)
    case(5);model%connectivity_csr=csr_from_dense(g,store_zeros=.true.)
    end select
    if(feature==0)then
      call model%fit(x)
    else
      if(allocated(features%connectivity))deallocate(features%connectivity)
      if(allocated(features%connectivity_mask))deallocate(features%connectivity_mask)
      if(allocated(features%connectivity_csr))deallocate(features%connectivity_csr)
      if(allocated(features%distance_threshold))deallocate(features%distance_threshold)
      features%n_clusters=model%n_clusters;features%compute_full_tree=model%compute_full_tree
      features%linkage=model%linkage;features%metric=model%metric
      features%compute_distances=model%compute_distances
      if(allocated(model%connectivity))features%connectivity=model%connectivity
      if(allocated(model%connectivity_mask))features%connectivity_mask=model%connectivity_mask
      if(allocated(model%connectivity_csr))features%connectivity_csr=model%connectivity_csr
      if(allocated(model%distance_threshold))features%distance_threshold=model%distance_threshold
      select case(pooling)
      case(0);features%pooling_func='mean'
      case(1);features%pooling_func='median'
      case(2);features%pooling_func='max'
      end select
      z=features%fit_transform(transpose(x));back=features%inverse_transform(z)
      model%labels_=features%labels_;model%children_=features%children_;model%distances_=features%distances_
      model%n_clusters_=features%n_clusters_;model%n_leaves_=features%n_leaves_
      model%n_features_in_=n;model%n_connected_components_=features%n_connected_components_
      model%compute_full_tree_=features%compute_full_tree_;model%connectivity_completed_=features%connectivity_completed_
    end if
    call require(all(x==copy),'complete/average linkage changed input')
    write(*,'(5(i0,1x),2(l1,1x))')model%n_clusters_,model%n_leaves_,model%n_features_in_, &
      model%n_connected_components_,size(model%children_,1),model%compute_full_tree_,model%connectivity_completed_
    write(*,'(*(i0,1x))')model%labels_
    do i=1,size(model%children_,1)
      write(*,'(2(i0,1x))')model%children_(i,:)
    end do
    write(*,'(i0)')size(model%distances_)
    write(*,'(*(es26.17e3,1x))')model%distances_
    if(feature/=0)then
      write(*,'(*(es26.17e3,1x))')z
      write(*,'(*(es26.17e3,1x))')back
    end if
    deallocate(x,g)
  end do
end program
