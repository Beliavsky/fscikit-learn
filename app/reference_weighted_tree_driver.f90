program reference_weighted_tree_driver
  use sklearn_common, only: dp
  use sklearn_tree
  implicit none
  class(decision_tree_classifier), allocatable :: clf
  class(decision_tree_regressor), allocatable :: reg
  real(dp), allocatable :: x(:,:),q(:,:),y(:),w(:),response(:,:)
  integer :: mode,n,p,m,depth,leaf,split,nf,seed,weighted,criterion,i,j,unit,out
  real(dp) :: decrease
  character(1024) :: path
  call get_command_argument(1,path); open(newunit=unit,file=trim(path),status='old')
  read(unit,*) mode,n,p,m,depth,leaf,split,nf,seed,weighted,criterion,decrease
  allocate(x(n,p),q(m,p),y(n),w(n))
  do i=1,n
    read(unit,*) x(i,:)
  end do
  read(unit,*) y; read(unit,*) w
  do i=1,m
    read(unit,*) q(i,:)
  end do
  close(unit)
  call get_command_argument(2,path); open(newunit=out,file=trim(path),status='replace')
  if(mod(mode,2)==0) then
    if(mode==0) then
      allocate(decision_tree_classifier::clf)
    else
      allocate(extra_tree_classifier::clf)
    end if
    clf%max_depth=depth; clf%min_samples_leaf=leaf; clf%min_samples_split=split
    clf%max_features=nf; clf%random_state=seed; clf%min_impurity_decrease=decrease
    if(criterion==1) clf%criterion='entropy'
    if(criterion==2) clf%criterion='log_loss'
    if(weighted==1) then
      call clf%fit(x,int(y),w)
    else
      call clf%fit(x,int(y))
    end if
    write(out,*) clf%node_count_,clf%max_features_,size(clf%classes_)
    write(out,*) clf%classes_
    write(out,*) clf%predict(q)
    response=clf%predict_proba(q)
    do i=1,m
      write(out,*) response(i,:)
    end do
    write(out,*) clf%feature_importances_
    do j=1,clf%node_count_
      associate(a=>clf%nodes_(j))
        write(out,*) a%left,a%right,a%feature,a%n_samples,a%depth,merge(1,0,a%missing_go_left)
        write(out,*) a%threshold,a%weighted_n_samples,a%impurity
        write(out,*) a%probability
      end associate
    end do
  else
    if(mode==1) then
      allocate(decision_tree_regressor::reg)
    else
      allocate(extra_tree_regressor::reg)
    end if
    reg%max_depth=depth; reg%min_samples_leaf=leaf; reg%min_samples_split=split
    reg%max_features=nf; reg%random_state=seed; reg%min_impurity_decrease=decrease
    if(weighted==1) then
      call reg%fit(x,y,w)
    else
      call reg%fit(x,y)
    end if
    write(out,*) reg%node_count_,reg%max_features_,0
    write(out,*) reg%predict(q)
    write(out,*) reg%feature_importances_
    do j=1,reg%node_count_
      associate(a=>reg%nodes_(j))
        write(out,*) a%left,a%right,a%feature,a%n_samples,a%depth,merge(1,0,a%missing_go_left)
        write(out,*) a%threshold,a%weighted_n_samples,a%impurity
        write(out,*) a%value
      end associate
    end do
  end if
  close(out)
end program
